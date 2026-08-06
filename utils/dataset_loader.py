import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================
# Dataset Path
# ==========================
DATASET_PATH = "dataset/real_vs_fake_realimg"

# ==========================
# Train Transform
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(3)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Validation/Test Transform
# ==========================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Datasets
# ==========================
train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, "train"),
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, "valid"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, "test"),
    transform=test_transform
)

# ==========================
# DataLoaders
# ==========================
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

classes = train_dataset.classes

print("Classes:", classes)
print("Training Images :", len(train_dataset))
print("Validation Images :", len(valid_dataset))
print("Testing Images :", len(test_dataset))