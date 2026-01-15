import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from training.dataset import FruitDatabase
from training.model import SimpleFruitClassifier
from training.train import train_model
from training.evaluate import evaluate_and_visualize

# Path to dataset and settings
DATA_DIR = "/Users/bertramsillesen/Desktop/archive/Fruit-262"   # HPC path
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 2

# Device processor
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset
full_dataset = FruitDatabase(DATA_DIR, transform=transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = SimpleFruitClassifier(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training settings
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS,
    save_path="trained_model.pt"
)

# Save the trained model
checkpoint = torch.load("model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)