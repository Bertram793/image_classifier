import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from training.dataset import FruitDatabase
from training.model import SimpleFruitClassifier
from training.evaluate import evaluate_and_visualize

# Fix for mac bug
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths to dataset and where trained model is stored
DATA_DIR = "/Users/bertramsillesen/Desktop/archive/Fruit-262"   # change if local
MODEL_PATH = "model.pt"
BATCH_SIZE = 32
NUM_CLASSES = 2

# Choose what processor it should run on
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# Transforms pictures to tensors and scale them
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the dataset
dataset = FruitDatabase(DATA_DIR, transform=transform)

test_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Model
model = SimpleFruitClassifier(num_classes=NUM_CLASSES).to(device)

# Load the trained model
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded from", MODEL_PATH)

# Analysis of the model
evaluate_and_visualize(
    model=model,
    dataloader=test_loader,
    dataset=dataset,
    device=device,
    max_images=16
)