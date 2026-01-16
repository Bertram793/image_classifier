# import nessecery liberies.
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
# _______________________________________________________________________________________________________________________________________________________
# NN-Setup hyper(parameters)
training_folder = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/Fruit_classification_data/train"
validation_folder = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/Fruit_classification_data/valid"
number_of_epochs = 100
learning_rate = 0.001
weight_decay = 0.0001
x_size, y_size = 32, 32
batch_size = 32
kernel_size = 3
patience = 3  # for early stopping

# from CPU to GPU
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
# _______________________________________________________________________________________________________________________________________________________

# Create dataset class, with pytorch Dataset and transform data.
# -> This automaticly names the pictures inside the different class folders.


class FruitDataset(Dataset):
    def __init__(self, dataset_folder):
        self.data = ImageFolder(dataset_folder, transform=transforms.Compose([
            transforms.Resize([x_size, y_size]),
            transforms.ToTensor()]))

    def __len__(self):
        # Print (len(dataset))  to see number of pictures in folder
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def classes(self):
        return self.data.classes


# Set up datasets for the training and validation loop
trainingset = FruitDataset(training_folder)
validation_set = FruitDataset(validation_folder)

# ______________________________________________________________________________________________________________________________________________________
# create dictionary for the different classes:
Classes = {}

for class_name, class_index in ImageFolder(training_folder).class_to_idx.items():
    # print(Classes) to see if all classes are correctly assigned.
    Classes[class_index] = class_name
# _________________________________________________________________________________________________________________________________________________________

# Create dataloader -> batch size can be adjusted in the "NN-setup section"

training_loader = DataLoader(trainingset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    validation_set, batch_size=batch_size, shuffle=False)
# _________________________________________________________________________________________________________________________________________________________

# simple neural network : arcitecture
neural_network = torch.nn.Sequential(
    # 3 inputs for "R,G,B" pictures
    nn.Conv2d(3, 32, kernel_size=kernel_size,
              padding=1),  # 3 inputs, 32 outputs
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=kernel_size,
              padding=1),  # 32 inputs, 64 outputs
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, kernel_size=kernel_size,
              padding=1),  # 64 inputs, 128 outputs
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(128, 256, kernel_size=kernel_size,
              padding=1),  # 128 inputs. 256 outputs
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),

    torch.nn.Flatten(),
    nn.Linear(256, 5),  # 256 inputs, 5 outputs


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
infinite = float("inf")
# --- Early Stopping setup)
best_validation_loss = infinite  # infite number
no_improvement = 0

# Training loop
for epoch in range(number_of_epochs):
    training_loss = 0
    correct_prediction = 0
    total_predictions = 0

    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = neural_network(images)
        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        predicted = outputs.argmax(1)
        correct_prediction += (predicted == labels).sum().item()
        total_predictions += len(labels)

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

    # --- Early Stopping check
    if average_validation_loss < best_validation_loss:

        best_validation_loss = average_validation_loss
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= patience:
        # Stop træningen
        print(f"Early stopping activated after {epoch+1} epochs.")
        break


# Plot loss functions
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("Picture resolution 32x32 : Loss over Epochs")
plt.legend()
plt.show()


# Testing the neural networks accuracy:

# establishing test folder
test_folder = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/Fruit_classification_data/test"


loader = DataLoader(ImageFolder(test_folder, transform=transforms.Compose([
    transforms.Resize([x_size, y_size]), transforms.ToTensor()])), batch_size=32)

neural_network.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in loader:
        predictions = neural_network(imgs.to(device)).argmax(1)
        correct += (predictions.cpu() == labels).sum().item()
        total += labels.size(0)

print(
    f"Accuracy on test-set: {100*correct/total:.2f}% ({correct}/{total} billeder korrekt)")

# Testing the neural networks accuracy: FINAL TEST

# establishing test folder
final_test_folder = "/Users/noahwestheimer/Documents/1. semester/ITIS 02461/ITIS PROJEKT/Fruits/resolution_test"


loader = DataLoader(ImageFolder(final_test_folder, transform=transforms.Compose([
    transforms.Resize([x_size, y_size]), transforms.ToTensor()])), batch_size=32)

neural_network.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in loader:
        predictions = neural_network(imgs.to(device)).argmax(1)
        correct += (predictions.cpu() == labels).sum().item()
        total += labels.size(0)

print(
    f"Accuracy on final_test-set: {100*correct/total:.2f}% ({correct}/{total} billeder korrekt)")
