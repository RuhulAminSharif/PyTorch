import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)

# check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")
print(train_df.shape)
print(test_df.shape)

# Create a 4x4 grid of images
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("First 16 Images", fontsize=16)

# # Plot the first 16 images from the dataset
# for i, ax in enumerate(axes.flat):
#     img = df.iloc[i, 1:].values.reshape(28, 28)  # Reshape to 28x28
#     ax.imshow(img)  # Display in grayscale
#     ax.axis("off")  # Remove axis for a cleaner look
#     ax.set_title(f"Label: {df.iloc[i, 0]}")  # Show the label

# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
# plt.show()

# train test split
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# scaling the feautures
X_train = X_train / 255.0
X_test = X_test / 255.0


# create CustomDataset Class
class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).reshape(
            -1, 1, 28, 28
        )  # batch size, channel, width, height
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# create train_dataset object
train_dataset = CustomDataset(X_train, y_train)

# create test_dataset object
test_dataset = CustomDataset(X_test, y_test)

# create train and test loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)


# define NN class
class MyNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=input_features,
                out_channels=32,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.4),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# set learning rate and epochs
epochs = 10
learning_rate = 0.1

# instatiate the model
model = MyNN(1)  # 1 channel for grayscale images
model = model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop
model.train()
for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        # move data to device
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # forward pass
        outputs = model(batch_features)

        # calculate loss
        loss = criterion(outputs, batch_labels)

        # back pass
        optimizer.zero_grad()
        loss.backward()

        # update grads
        optimizer.step()

        total_epoch_loss = total_epoch_loss + loss.item()

    avg_loss = total_epoch_loss / len(train_loader)
    print(f"Epoch: {epoch + 1} , Loss: {avg_loss}")

# set model to eval mode
model.eval()

# evaluation code
total = 0
correct = 0

with torch.no_grad():

    for batch_features, batch_labels in test_loader:
        # move data to device
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        total = total + batch_labels.shape[0]

        correct = correct + (predicted == batch_labels).sum().item()

print(correct / total)
