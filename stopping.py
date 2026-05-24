"""
stopping.py
=================
Step 1 – Adaptive Early Stopping with Checkpoint Rolling Restoration
---------------------------------------------------------------------
Motivation
----------
Trying to prevent memorisation and underfitting...and show we're applying the course theory.

    Patience window  P  – how many consecutive "non-improving" epochs we
                          tolerate before halting.  A window rather than a
                          single-step trigger accounts for the plateau-and-dip
                          dynamics we saw in the original fixed epoch runs (loss
                          temporarily rises while the optimiser escapes a saddle 
                          pt).

    Delta threshold  epsilon  – the minimum meaningful improvement in validation
                                loss.  Improvements smaller than epsilon are treated as
                                numerical noise, not genuine progress.  This prevents
                                the window counter from resetting on trivially small
                                fluctuations.

    Restore best     –   Keep track of the best version and restore it. Unavoidable
                         because of the patience window.

Stopping criterion
---------------------------
Let  L = min( L_val(0), …, L_val(t-1) )  be the running historical minimum.

At epoch t, the counter increments if:

    L_val(t)  ≥  L − epsilon        ...i.e. no meaningful improvement

Otherwise the counter resets to 0 and L is updated.

When the counter reaches P, training terminates and the best checkpoint is
restored.
"""

import copy
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


#EarlyStopping

class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        delta: float = 1e-4,
        checkpoint_path: Optional[str | Path] = None,
        verbose: bool = True,
        mode: str = "min",
    ) -> None:
        #Argument validation
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

        #Internal state
        #Initialise best score to the worst possible value for each mode so
        #that the very first epoch always registers as an improvement.
        self._best_score: float = math.inf if mode == "min" else -math.inf
        self._counter:    int   = 0
        self._best_state: Optional[dict] = None   #CPU copy of state_dict
        self._stopped:    bool  = False
        self._best_epoch: int   = 0
        self._epoch:      int   = 0               #tracks calls to .step()

    #Read-only properties

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    #Core logic

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self._best_score - self.delta
        else:
            return metric > self._best_score + self.delta

    def _save_checkpoint(self, model: nn.Module) -> None:
        #Move all tensors to CPU before storing — ensures portability
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
            #deepcopy: guarantees isolation — subsequent parameter updates
            #cannot mutate the cached tensors.
            self._best_state = copy.deepcopy(cpu_state)
            if self.verbose:
                log.info("[EarlyStopping] Best checkpoint cached in RAM.")

    def step(self, metric: float, model: nn.Module) -> bool:
        if self._stopped:
            #Guard: do not advance state after stopping has been triggered.
            log.warning(
                "[EarlyStopping] .step() called after stopping was triggered. "
                "Ignoring."
            )
            return True

        self._epoch += 1

        if self._is_improvement(metric):
            #Genuine improvement
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
            self._best_epoch = self._epoch - 1   #convert to 0-based
            self._counter    = 0
            self._save_checkpoint(model)

        else:
            #No meaningful improvement
            self._counter += 1
            if self.verbose:
                log.info(
                    "[EarlyStopping] Epoch %d  │  no improvement "
                    "(%.6f vs best %.6f, δ=%.1e)  │  counter %d / %d.",
                    self._epoch, metric, self._best_score,
                    self.delta, self._counter, self.patience,
                )

            if self._counter >= self.patience:
                #Patience exhausted
                self._stopped = True
                if self.verbose:
                    log.info(
                        "[EarlyStopping] Patience (%d) exhausted at epoch %d. "
                        "Best score %.6f at epoch %d. Restoring best weights.",
                        self.patience, self._epoch,
                        self._best_score, self._best_epoch,
                    )
                self.load_best(model)
                return True   #signal the training loop to break

        return False   #continue training

    def load_best(self, model: nn.Module) -> None:
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



#Integration helper — drop-in training loop wrapper

def run_training_loop(
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler,                            #any LR scheduler with .step()
    train_fn,                             #callable(model, optimizer) → (loss, acc)
    eval_fn,                              #callable(model) → (loss, acc)
    max_epochs:   int   = 100,
    patience:     int   = 5,
    delta:        float = 1e-4,
    checkpoint:   Optional[str | Path] = "Models/best.pt",
    verbose:      bool  = True,
) -> dict:
    #Configure logging so EarlyStopping messages reach stdout
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
        mode="min",   #monitoring validation LOSS
    )

    history: dict = {
        "train_loss":   [],
        "train_acc":    [],
        "val_loss":     [],
        "val_acc":      [],
        "best_epoch":   0,
        "stopped_early": False,
    }

    #Table header
    hdr = (
        f"{'Epoch':>6}  {'Tr Loss':>9}  {'Tr Acc':>8}  "
        f"{'Va Loss':>9}  {'Va Acc':>8}  "
        f"{'LR':>10}  {'ES Counter':>10}"
    )
    print(hdr)
    print("─" * len(hdr))

    for epoch in range(max_epochs):

        #1.Forward + backward pass over the training set
        #optimizer.zero_grad(), loss.backward(), optimizer.step() happen
        #inside train_fn — kept there so each architecture can own its
        #own gradient-flow logic (e.g. gradient clipping for ResNet).
        tr_loss, tr_acc = train_fn(model, optimizer)

        #2.Validation pass
        vl_loss, vl_acc = eval_fn(model)

        #3.LR schedule tick
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        #4.History bookkeeping
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        #5. Early stopping evaluation
        #stopper.step() snapshots the model if val_loss improved,
        #increments the counter otherwise, and returns True when patience
        #is exhausted.
        should_stop = stopper.step(vl_loss, model)

        #6. Epoch log line
        marker = " ★" if stopper.counter == 0 else f"  {stopper.counter}/{patience}"
        print(
            f"{epoch+1:>6}  {tr_loss:>9.5f}  {tr_acc:>7.2f}%  "
            f"{vl_loss:>9.5f}  {vl_acc:>7.2f}%  "
            f"{current_lr:>10.2e}  {marker}"
        )

        if should_stop:
            history["stopped_early"] = True
            print(
                f"\n  Early stop triggered at epoch {epoch + 1}.  "
                f"Best val loss {stopper.best_score:.6f} "
                f"at epoch {stopper.best_epoch + 1}.\n"
            )
            break

    #7. Always restore best — even if max_epochs was reached
    #This guarantees the returned model holds the best checkpoint, not the final (potentially overfitted) epoch.
    stopper.load_best(model)
    history["best_epoch"] = stopper.best_epoch

    return history



#Quick self-test


if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    print("=" * 60)
    print("  EarlyStopping self-test")
    print("=" * 60)

    #Tiny toy network — just to have a real state_dict to checkpoint
    toy_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    stopper = EarlyStopping(
        patience=3,
        delta=1e-3,
        checkpoint_path=None,   #RAM only for the test
        verbose=True,
        mode="min",
    )

    #Simulate a loss trajectory:
    #epochs 1-5: improving, epoch 6-9: plateau (should trigger at epoch 9)
    simulated_losses = [0.90, 0.75, 0.60, 0.52, 0.48,   #improving
                        0.49, 0.50, 0.49, 0.50]           #plateau => stop

    print(f"\nSimulated val losses: {simulated_losses}")
    print(f"Patience={stopper.patience}, delta={stopper.delta}\n")

    for ep, loss in enumerate(simulated_losses, start=1):
        #Mutate toy model weights so each checkpoint is distinct
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
