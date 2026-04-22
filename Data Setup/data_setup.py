import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# Paths to your dataset
# =========================
train_dir = r"C:\Users\raiha\Downloads\MRI Tumor Dataset\split_70_15_15\train"
val_dir = r"C:\Users\raiha\Downloads\MRI Tumor Dataset\split_70_15_15\val"
test_dir = r"C:\Users\raiha\Downloads\MRI Tumor Dataset\split_70_15_15\test"

# =========================
# Image transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# =========================
# Load datasets
# =========================
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

# =========================
# Create dataloaders
# =========================
batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# Basic checks
# =========================
print("Class names:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
print("Number of test images:", len(test_dataset))

images, labels = next(iter(train_loader))
print("One batch image shape:", images.shape)
print("One batch label shape:", labels.shape)