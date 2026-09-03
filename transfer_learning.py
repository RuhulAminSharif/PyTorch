import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms, InterpolationMode
import optuna
from PIL import Image
import numpy as np

torch.manual_seed(42)

# check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv("fmnist_small.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")
print(train_df.shape)
print(test_df.shape)

# train test split
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# transforming the features
custom_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# create CustomDataset Class
class CustomDataset(Dataset):

    def __init__(self, features, labels, transform):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # resize to (28, 28)
        image = self.features[index].reshape(28, 28)

        # change datatype to np.uint8
        image = image.astype(np.uint8)

        # change black&white to color ->
        # (C,H,W) -> (H,W,C)
        image = np.stack([image] * 3, axis=-1)

        # convert array to PIL image
        image = Image.fromarray(image)

        # apply transforms
        image = self.transform(image)

        # return
        return image, torch.tensor(self.labels[index], dtype=torch.long)


# create train_dataset object
train_dataset = CustomDataset(X_train, y_train, transform=custom_transform)

# create test_dataset object
test_dataset = CustomDataset(X_test, y_test, transform=custom_transform)

# create train and test loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)


# fetch the pretrained model
import torchvision.models as models

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# freeze the parameters of the pretrained model's feature extractor
for param in vgg16.features.parameters():
    param.requires_grad = False

# define the classifier for the pretrained model
# replacing existing classifier with a new one for our specific task
vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10),
)

vgg16 = vgg16.to(device)

learning_rate = 0.0001
epochs = 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=learning_rate)

# training loop
for epoch in range(epochs):
    vgg16.train()
    running_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(
            device
        )

        outputs = vgg16(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# validation loop
vgg16.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(
            device
        )
        outputs = vgg16(batch_features)
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
