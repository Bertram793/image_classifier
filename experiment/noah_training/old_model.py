# import nessecery liberies.
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
# _______________________________________________________________________________________________________________________________________________________
# Setup for parameters:
training_folder = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/Fruit_classification_data/train"
validation_folder = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/Fruit_classification_data/valid"
number_of_epochs = 50
learning_rate = 0.0001
weight_decay = 0.0001
x_size, y_size = 128, 128
batch_size = 32
kernel_size = 3


# USE GPU
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
# _______________________________________________________________________________________________________________________________________________________

# Create dataset class, with pytorch Dataset and transform data.


class FruitDataset(Dataset):
    def __init__(self, dataset_folder):
        self.data = ImageFolder(dataset_folder, transform=transforms.Compose([
            transforms.Resize([x_size, y_size]),
            transforms.ToTensor()]))

    def __len__(self):
        return len(self.data)  # Print(len(dataset)) to see number of pictures

    def __getitem__(self, idx):
        return self.data[idx]

    def classes(self):
        return self.data.classes


# datasets
trainingset = FruitDataset(training_folder)
validation_set = FruitDataset(validation_folder)

# ______________________________________________________________________________________________________________________________________________________


# create dictionary for the different classes:
Classes = {}

for class_name, class_index in ImageFolder(training_folder).class_to_idx.items():
    # print(Classes) to see if all classes are correctly assigned.
    Classes[class_index] = class_name
# _________________________________________________________________________________________________________________________________________________________

# Create dataloader

training_loader = DataLoader(trainingset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    validation_set, batch_size=batch_size, shuffle=False)


# _________________________________________________________________________________________________________________________________________________________
neural_network = torch.nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=kernel_size, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    nn.Linear(128, 5),


).to(device)
# ____________________________________________________________________________________________________________________________________________________________


# loss

Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    neural_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
# ____________________________________________________________________________________________________________________________________________________________

# collect train and val loss for loss functions
train_loss = []
val_loss = []


# Training loop
for epoch in range(number_of_epochs):
    training_loss = 0
    correct_prediction = 0     # uændret navn
    total_predictions = 0      # uændret navn

    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = neural_network(images)
        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        # --- NY SIMPLERE ACCURACY ---
        predicted = outputs.argmax(1)  # enklere end torch.max
        correct_prediction += (predicted == labels).sum().item()
        total_predictions += len(labels)
        # ----------------------------

    average_train_loss = training_loss / len(training_loader)
    train_loss.append(average_train_loss)

    training_accuracy = 100 * correct_prediction / total_predictions

    # Validation loop
    neural_network.eval()
    validation_loss = 0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = neural_network(images)
            loss = Loss(outputs, labels)
            validation_loss += loss.item()

            predicted = outputs.argmax(1)
            validation_correct += (predicted == labels).sum().item()
            validation_total += len(labels)

    average_validation_loss = validation_loss / len(validation_loader)
    val_loss.append(average_validation_loss)

    validation_accuracy = 100 * validation_correct / validation_total

    print(f"{epoch+1} / {number_of_epochs}, Training loss = {average_train_loss:.3f} Training Accuracy = {training_accuracy:.3f} ")
    print("______________________________________ \n")
    print(
        f"Validation loss = {average_validation_loss:.3f} Validation Accuracy = {validation_accuracy:.3f}")
    print("______________________________________ \n")


# Plot loss functions
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.show()


# Funktion til at forudsige et enkelt billede

def predict_image(image_path, model, class_dict):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize([x_size, y_size]),
        transforms.ToTensor()
    ])

    # Indlæs billede
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # batch dimension

    # Model prediction
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = output.argmax(1).item()

    print(f"Predicted class: {class_dict[predicted_class]}")
    return class_dict[predicted_class]


# EKSEMPEL PÅ BRUG:
# <-- indsæt dit billede her
image_path = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/IMG_4519.jpeg"
prediction = predict_image(image_path, neural_network, Classes)
print("Model prediction:", prediction)


#