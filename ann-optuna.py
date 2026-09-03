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

train_df = pd.read_csv("fmnist_small.csv")
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
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# create train_dataset object
train_dataset = CustomDataset(X_train, y_train)

# create test_dataset object
test_dataset = CustomDataset(X_test, y_test)


class MyNN(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate
    ):
        super().__init__()
        layers = []
        current_input_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_input_dim, neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            current_input_dim = neurons_per_layer

        layers.append(nn.Linear(current_input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


# objective function
def objective_function(trial):
    # next hyperparameter values from the search space
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    neurons_per_layer = trial.suggest_int("neurons_per_layer", 8, 128, step=8)
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # create train and test loader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # model initialization
    input_dim = 784
    output_dim = 10

    model = MyNN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
        neurons_per_layer=neurons_per_layer,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    # training loop
    for _ in range(epochs):
        model.train()
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

    accuracy = correct / total

    return accuracy


import optuna

study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective_function, n_trials=10)

print(study.best_value)
print(study.best_params)
