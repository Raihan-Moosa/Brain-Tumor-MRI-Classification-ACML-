"""
early_stopping.py
=================
Step 1 – Adaptive Early Stopping with Checkpoint Rolling Restoration
---------------------------------------------------------------------
Motivation
----------
Fixed epoch limits (10 / 15 / 40) are arbitrary. Two failure modes result:

  * Underfitting    – training ends before the loss landscape is fully explored.
  * Memorisation    – training continues through the generalisation optimum and
                      the model absorbs dataset-specific noise.

This module implements a rigorous, theory-grounded stopping criterion with
three independently configurable axes:

    Patience window  P  – how many consecutive "non-improving" epochs we
                          tolerate before halting.  A window rather than a
                          single-step trigger accounts for the plateau-and-dip
                          dynamics common in deep networks (loss temporarily
                          rises while the optimiser escapes a saddle point).

    Delta threshold  ε  – the minimum meaningful improvement in validation
                          loss.  Improvements smaller than ε are treated as
                          numerical noise, not genuine progress.  This prevents
                          the window counter from resetting on trivially small
                          fluctuations.

    Restore best     –   On stopping (or on explicit .load_best()), the module
                         re-loads the state_dict captured at the historical
                         minimum, NOT the state from the final, overfitted epoch.

Stopping criterion (formal)
---------------------------
Let  L̂ = min( L_val(0), …, L_val(t-1) )  be the running historical minimum.

At epoch t the counter increments if:

    L_val(t)  ≥  L̂ − ε          … i.e. no meaningful improvement

Otherwise the counter resets to 0 and L̂ is updated.

When the counter reaches P, training terminates and the best checkpoint is
restored.

Integration with your existing scripts
---------------------------------------
Drop-in replacement for the ad-hoc early-stopping logic found in
train_improved.py and BrainTumorNet-Lite. Replace the manual `no_improve`
counter block with:

    stopper = EarlyStopping(patience=5, delta=1e-4,
                            checkpoint_path="Models/best.pt")
    for epoch in range(MAX_EPOCHS):
        train_one_epoch(...)
        val_loss, val_acc = evaluate(...)
        if stopper.step(val_loss, model):
            break
    stopper.load_best(model)   # always restore, whether stopped early or not

Usage – accuracy-mode (mode="max")
------------------------------------
    stopper = EarlyStopping(patience=5, delta=5e-3, mode="max")
    if stopper.step(val_acc, model):
        ...

Dependencies: torch (stdlib only otherwise)
"""

import copy
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  EarlyStopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Validation-metric monitor with patience window and automatic checkpoint
    rolling restoration.

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving epochs to tolerate before
        triggering a stop.  Recommended: 5–10 for medical imaging tasks.

    delta : float
        Minimum change in the monitored metric that qualifies as an
        improvement.  Acts as a noise floor — changes smaller than delta
        are ignored.  Typical values: 1e-4 (loss) or 5e-3 (accuracy).

    checkpoint_path : str or Path, optional
        Filesystem path at which the best state_dict is written with
        torch.save().  If None the checkpoint is held in CPU RAM only via
        copy.deepcopy — adequate for short runs but loses state on crash.
        Providing a path is strongly recommended for long training jobs.

    verbose : bool
        Emit INFO-level log lines on every improvement and patience event.

    mode : {"min", "max"}
        "min"  → lower metric is better  (validation loss, default)
        "max"  → higher metric is better (validation accuracy, F1, …)

    Attributes (read-only properties)
    ----------------------------------
    stopped      : bool  – True once patience is exhausted.
    best_score   : float – Best metric value seen so far.
    counter      : int   – Current consecutive non-improving epoch count.
    best_epoch   : int   – 0-based epoch index where the best score occurred.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 1e-4,
        checkpoint_path: Optional[str | Path] = None,
        verbose: bool = True,
        mode: str = "min",
    ) -> None:
        # ── Argument validation ───────────────────────────────────────────────
        if patience < 1:
            raise ValueError(f"patience must be ≥ 1, got {patience}.")
        if delta < 0:
            raise ValueError(f"delta must be ≥ 0, got {delta}.")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.patience         = patience
        self.delta            = delta
        self.checkpoint_path  = Path(checkpoint_path) if checkpoint_path else None
        self.verbose          = verbose
        self.mode             = mode

        # ── Internal state ────────────────────────────────────────────────────
        # Initialise best score to the worst possible value for each mode so
        # that the very first epoch always registers as an improvement.
        self._best_score: float = math.inf if mode == "min" else -math.inf
        self._counter:    int   = 0
        self._best_state: Optional[dict] = None   # CPU copy of state_dict
        self._stopped:    bool  = False
        self._best_epoch: int   = 0
        self._epoch:      int   = 0               # tracks calls to .step()

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def stopped(self) -> bool:
        """True once the patience window has been exhausted."""
        return self._stopped

    @property
    def best_score(self) -> float:
        """The historical best metric value seen so far."""
        return self._best_score

    @property
    def counter(self) -> int:
        """Current number of consecutive non-improving epochs."""
        return self._counter

    @property
    def best_epoch(self) -> int:
        """0-based epoch index at which the best score was recorded."""
        return self._best_epoch

    # ── Core logic ────────────────────────────────────────────────────────────

    def _is_improvement(self, metric: float) -> bool:
        """
        Return True if `metric` represents a meaningful improvement over the
        stored best score.

        The comparison respects both `mode` and `delta`:

            mode="min":  improvement iff  metric  <  best_score − delta
            mode="max":  improvement iff  metric  >  best_score + delta

        The delta term enforces a minimum bar — tiny fluctuations that fall
        within the noise floor do NOT reset the patience counter.
        """
        if self.mode == "min":
            return metric < self._best_score - self.delta
        else:
            return metric > self._best_score + self.delta

    def _save_checkpoint(self, model: nn.Module) -> None:
        """
        Persist the model's current state_dict.

        Two storage paths:
          1. checkpoint_path is set  → torch.save to disk (survives crashes).
          2. checkpoint_path is None → copy.deepcopy into RAM only.

        The state_dict is explicitly moved to CPU before saving so the
        checkpoint is always device-agnostic and can be loaded on CPU-only
        machines regardless of whether training used a GPU.
        """
        # Move all tensors to CPU before storing — ensures portability
        cpu_state = {
            k: v.cpu() for k, v in model.state_dict().items()
        }

        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cpu_state, self.checkpoint_path)
            if self.verbose:
                log.info(
                    "[EarlyStopping] Best checkpoint saved → %s",
                    self.checkpoint_path,
                )
        else:
            # deepcopy: guarantees isolation — subsequent parameter updates
            # cannot mutate the cached tensors.
            self._best_state = copy.deepcopy(cpu_state)
            if self.verbose:
                log.info("[EarlyStopping] Best checkpoint cached in RAM.")

    def step(self, metric: float, model: nn.Module) -> bool:
        """
        Evaluate one epoch's validation metric and update stopping state.

        Call once per epoch, **after** the validation loop completes.

        Parameters
        ----------
        metric : float
            Validation loss (mode="min") or accuracy/F1 (mode="max") for
            the current epoch.
        model  : nn.Module
            The model being trained.  Its state_dict is snapshotted whenever
            a new best score is recorded.

        Returns
        -------
        bool
            True  → patience exhausted; training should halt.
                    The best checkpoint has already been restored onto `model`.
            False → training should continue.

        Algorithm
        ---------
        1. Check whether `metric` constitutes a meaningful improvement via
           _is_improvement (respects `mode` and `delta`).

        2a. Improvement detected:
              • Update _best_score.
              • Reset _counter to 0.
              • Snapshot model via _save_checkpoint.

        2b. No improvement:
              • Increment _counter by 1.
              • If _counter reaches `patience`:
                  – Set _stopped = True.
                  – Call load_best(model) to restore the historical best weights.
                  – Return True (signal to break the training loop).

        3. Return False (continue training).
        """
        if self._stopped:
            # Guard: do not advance state after stopping has been triggered.
            log.warning(
                "[EarlyStopping] .step() called after stopping was triggered. "
                "Ignoring."
            )
            return True

        self._epoch += 1

        if self._is_improvement(metric):
            # ── Genuine improvement ───────────────────────────────────────────
            delta_str = (
                f"{abs(metric - self._best_score):.6f}"
                if not math.isinf(self._best_score)
                else "—"
            )
            if self.verbose:
                log.info(
                    "[EarlyStopping] Epoch %d  │  %s improved: "
                    "%.6f → %.6f  (Δ %s)  │  counter reset.",
                    self._epoch,
                    "loss" if self.mode == "min" else "metric",
                    self._best_score if not math.isinf(self._best_score) else float("nan"),
                    metric,
                    delta_str,
                )
            self._best_score = metric
            self._best_epoch = self._epoch - 1   # convert to 0-based
            self._counter    = 0
            self._save_checkpoint(model)

        else:
            # ── No meaningful improvement ─────────────────────────────────────
            self._counter += 1
            if self.verbose:
                log.info(
                    "[EarlyStopping] Epoch %d  │  no improvement "
                    "(%.6f vs best %.6f, δ=%.1e)  │  counter %d / %d.",
                    self._epoch, metric, self._best_score,
                    self.delta, self._counter, self.patience,
                )

            if self._counter >= self.patience:
                # ── Patience exhausted ────────────────────────────────────────
                self._stopped = True
                if self.verbose:
                    log.info(
                        "[EarlyStopping] Patience (%d) exhausted at epoch %d. "
                        "Best score %.6f at epoch %d. Restoring best weights.",
                        self.patience, self._epoch,
                        self._best_score, self._best_epoch,
                    )
                self.load_best(model)
                return True   # ← signal the training loop to break

        return False   # ← continue training

    def load_best(self, model: nn.Module) -> None:
        """
        Restore the best-ever state_dict onto `model`.

        This is idempotent — safe to call multiple times and safe to call
        at the end of a training loop even if early stopping was never
        triggered (ensures the final deployed model always holds the best
        checkpoint, not the last epoch's weights).

        Parameters
        ----------
        model : nn.Module
            The model to restore.  The checkpoint is loaded with
            strict=True so any architecture mismatch raises immediately
            rather than silently loading a partial state.

        Raises
        ------
        RuntimeError
            If no checkpoint has been saved yet (i.e. .step() was never
            called with an improving metric).
        """
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location="cpu",
                               weights_only=True)
            model.load_state_dict(state, strict=True)
            if self.verbose:
                log.info(
                    "[EarlyStopping] Restored best weights from disk: %s "
                    "(best epoch %d, score %.6f).",
                    self.checkpoint_path, self._best_epoch, self._best_score,
                )

        elif self._best_state is not None:
            model.load_state_dict(self._best_state, strict=True)
            if self.verbose:
                log.info(
                    "[EarlyStopping] Restored best weights from RAM "
                    "(best epoch %d, score %.6f).",
                    self._best_epoch, self._best_score,
                )

        else:
            raise RuntimeError(
                "EarlyStopping.load_best() called before any checkpoint was "
                "saved.  Ensure .step() is called at least once with a valid "
                "metric before calling .load_best()."
            )

    def state_dict(self) -> dict:
        """
        Serialise the monitor's internal state for resumable training.

        Returns a plain dict that can be saved alongside the model checkpoint
        and passed to load_state_dict() to resume monitoring across sessions.

        Example
        -------
            torch.save({
                "model":   model.state_dict(),
                "stopper": stopper.state_dict(),
            }, "checkpoint.pt")
        """
        return {
            "patience":        self.patience,
            "delta":           self.delta,
            "mode":            self.mode,
            "best_score":      self._best_score,
            "counter":         self._counter,
            "stopped":         self._stopped,
            "best_epoch":      self._best_epoch,
            "epoch":           self._epoch,
            "checkpoint_path": str(self.checkpoint_path)
                               if self.checkpoint_path else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore monitor state from a previously serialised state_dict.

        Does NOT restore the model weights themselves — call load_best()
        separately for that.

        Parameters
        ----------
        state : dict
            Dict returned by a previous call to .state_dict().
        """
        self.patience         = state["patience"]
        self.delta            = state["delta"]
        self.mode             = state["mode"]
        self._best_score      = state["best_score"]
        self._counter         = state["counter"]
        self._stopped         = state["stopped"]
        self._best_epoch      = state["best_epoch"]
        self._epoch           = state["epoch"]
        self.checkpoint_path  = (
            Path(state["checkpoint_path"])
            if state["checkpoint_path"] else None
        )

    def summary(self) -> str:
        """
        Return a human-readable one-line summary of the current monitor state.
        Useful for embedding in epoch log lines.

        Example output
        --------------
        EarlyStopping | mode=min  patience=5  delta=1e-04 |
        best=0.123456 @ epoch 7  counter=2/5  status=running
        """
        status = "STOPPED" if self._stopped else "running"
        return (
            f"EarlyStopping | mode={self.mode}  patience={self.patience}  "
            f"delta={self.delta:.0e} | "
            f"best={self._best_score:.6f} @ epoch {self._best_epoch}  "
            f"counter={self._counter}/{self.patience}  status={status}"
        )

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, delta={self.delta}, "
            f"mode='{self.mode}', checkpoint_path={self.checkpoint_path!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Integration helper — drop-in training loop wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_training_loop(
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler,                            # any LR scheduler with .step()
    train_fn,                             # callable(model, optimizer) → (loss, acc)
    eval_fn,                              # callable(model) → (loss, acc)
    max_epochs:   int   = 100,
    patience:     int   = 5,
    delta:        float = 1e-4,
    checkpoint:   Optional[str | Path] = "Models/best.pt",
    verbose:      bool  = True,
) -> dict:
    """
    Transparent training loop with EarlyStopping wired in.

    This is the explicit manual loop referenced in the project brief —
    zero high-level trainer abstractions.  Every gradient step, scheduler
    tick, and stopping decision is visible and auditable.

    Parameters
    ----------
    model        : The nn.Module being trained.
    optimizer    : Configured optimiser (AdamW, SGD, …).
    scheduler    : LR scheduler; .step() is called once per epoch.
    train_fn     : Callable that runs one full training epoch and returns
                   (train_loss: float, train_acc: float).
    eval_fn      : Callable that runs validation and returns
                   (val_loss: float, val_acc: float).
    max_epochs   : Hard upper bound on training epochs.
    patience     : Passed to EarlyStopping.
    delta        : Passed to EarlyStopping.
    checkpoint   : Path for best-weight persistence.
    verbose      : Forward to EarlyStopping; also prints epoch table rows.

    Returns
    -------
    history : dict with keys
        "train_loss", "train_acc", "val_loss", "val_acc"  – list per epoch.
        "best_epoch"  – index of the epoch with the best validation loss.
        "stopped_early" – bool.
    """
    # Configure logging so EarlyStopping messages reach stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    stopper = EarlyStopping(
        patience=patience,
        delta=delta,
        checkpoint_path=checkpoint,
        verbose=verbose,
        mode="min",   # monitoring validation LOSS
    )

    history: dict = {
        "train_loss":   [],
        "train_acc":    [],
        "val_loss":     [],
        "val_acc":      [],
        "best_epoch":   0,
        "stopped_early": False,
    }

    # Table header
    hdr = (
        f"{'Epoch':>6}  {'Tr Loss':>9}  {'Tr Acc':>8}  "
        f"{'Va Loss':>9}  {'Va Acc':>8}  "
        f"{'LR':>10}  {'ES Counter':>10}"
    )
    print(hdr)
    print("─" * len(hdr))

    for epoch in range(max_epochs):

        # ── 1. Forward + backward pass over the training set ─────────────────
        #   optimizer.zero_grad(), loss.backward(), optimizer.step() happen
        #   inside train_fn — kept there so each architecture can own its
        #   own gradient-flow logic (e.g. gradient clipping for ResNet).
        tr_loss, tr_acc = train_fn(model, optimizer)

        # ── 2. Validation pass (no_grad) ──────────────────────────────────────
        vl_loss, vl_acc = eval_fn(model)

        # ── 3. LR schedule tick ───────────────────────────────────────────────
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── 4. History bookkeeping ────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        # ── 5. Early stopping evaluation ─────────────────────────────────────
        #   stopper.step() snapshots the model if val_loss improved,
        #   increments the counter otherwise, and returns True when patience
        #   is exhausted.
        should_stop = stopper.step(vl_loss, model)

        # ── 6. Epoch log line ─────────────────────────────────────────────────
        marker = " ★" if stopper.counter == 0 else f"  {stopper.counter}/{patience}"
        print(
            f"{epoch+1:>6}  {tr_loss:>9.5f}  {tr_acc:>7.2f}%  "
            f"{vl_loss:>9.5f}  {vl_acc:>7.2f}%  "
            f"{current_lr:>10.2e}  {marker}"
        )

        if should_stop:
            history["stopped_early"] = True
            print(
                f"\n  ⚑  Early stop triggered at epoch {epoch + 1}.  "
                f"Best val loss {stopper.best_score:.6f} "
                f"at epoch {stopper.best_epoch + 1}.\n"
            )
            break

    # ── 7. Always restore best — even if max_epochs was reached ──────────────
    #   This guarantees the returned model holds the best checkpoint, not
    #   the final (potentially overfitted) epoch.
    stopper.load_best(model)
    history["best_epoch"] = stopper.best_epoch

    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test (python early_stopping.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    print("=" * 60)
    print("  EarlyStopping self-test")
    print("=" * 60)

    # Tiny toy network — just to have a real state_dict to checkpoint
    toy_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    stopper = EarlyStopping(
        patience=3,
        delta=1e-3,
        checkpoint_path=None,   # RAM only for the test
        verbose=True,
        mode="min",
    )

    # Simulate a loss trajectory:
    #   epochs 1-5: improving, epoch 6-9: plateau (should trigger at epoch 9)
    simulated_losses = [0.90, 0.75, 0.60, 0.52, 0.48,   # improving
                        0.49, 0.50, 0.49, 0.50]           # plateau → stop

    print(f"\nSimulated val losses: {simulated_losses}")
    print(f"Patience={stopper.patience}, delta={stopper.delta}\n")

    for ep, loss in enumerate(simulated_losses, start=1):
        # Mutate toy model weights so each checkpoint is distinct
        with torch.no_grad():
            for p in toy_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        triggered = stopper.step(loss, toy_model)
        print(f"  Epoch {ep:>2}  loss={loss:.4f}  {stopper.summary()}")

        if triggered:
            print(f"\n  → stop triggered at epoch {ep}.")
            break

    stopper.load_best(toy_model)
    print(f"\n  Best score : {stopper.best_score}  (epoch {stopper.best_epoch + 1})")
    print(f"  State dict : {stopper.state_dict()}")
    print("\n  Self-test complete.\n")