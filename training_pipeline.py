import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================================================================
# 1. DATA LOADING & PREPARATION
# ==============================================================================

# Load Breast Cancer dataset from a public repository
data_url = "https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv"
df = pd.read_csv(data_url)

# Drop irrelevant columns (ID and empty padding column)
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Separate features (X) and target labels (y)
# Target is column 0 ('diagnosis'), Features are columns 1 to end
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================================
# 2. DATA PREPROCESSING
# ==============================================================================
# Standardize features (mean=0, variance=1) for faster, stable convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Used transform() to avoid data leakage

# Encode categorical labels ('M'/'B') to numerical (1/0)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Convert NumPy arrays to PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train_scaled).to(torch.float64)
X_test_tensor = torch.from_numpy(X_test_scaled).to(torch.float64)

# Reshape target tensors from (N,) to (N, 1) to match network output shape
y_train_tensor = torch.from_numpy(y_train_encoded).view(-1, 1).to(torch.float64)
y_test_tensor = torch.from_numpy(y_test_encoded).view(-1, 1).to(torch.float64)


# ==============================================================================
# 3. MODEL DEFINITION (Single Layer Neural Network)
# ==============================================================================
class SimpleNN:
    def __init__(self, input_features) -> None:
        """
        Initializes weights and biases.
        requires_grad=True tells PyTorch to track operations for backpropagation.
        """
        # Weights shape: (number_of_features, 1)
        self.weights = torch.rand(
            input_features.shape[1], 1, dtype=torch.float64, requires_grad=True
        )
        # Bias shape: (1)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def forward(self, X):
        """
        Forward pass: Computes the linear combination and applies Sigmoid activation.
        z = XW + b
        y_pred = 1 / (1 + e^-z)
        """
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y_true):
        """
        Computes Binary Cross Entropy (BCE) Loss.
        """
        # Clamp predictions to [epsilon, 1-epsilon] to avoid torch.log(0) resulting in NaN
        epsilon = 1e-7
        y_pred_clamped = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Vectorized BCE calculation
        loss = -(
            y_true * torch.log(y_pred_clamped)
            + (1 - y_true) * torch.log(1 - y_pred_clamped)
        ).mean()
        return loss


# ==============================================================================
# 4. TRAINING PIPELINE
# ==============================================================================
learning_rate = 0.01
epochs = 50

# Instantiate the model
model = SimpleNN(X_train_tensor)

print("Starting Training Loop...\n" + "-" * 30)

for epoch in range(epochs):
    # 1. Forward Pass: Compute predictions based on current weights
    y_pred = model.forward(X_train_tensor)

    # 2. Loss Calculation: Measure error against actual targets
    loss = model.loss_function(y_pred, y_train_tensor)

    # 3. Backward Pass: Compute gradients (dL/dw, dL/db) automatically
    loss.backward()

    # 4. Parameter Update: Adjust weights using Gradient Descent
    # Wrap in torch.no_grad() because we don't want to track these update operations
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    # 5. Zero Gradients: Clear the accumulated gradients for the next epoch
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch: {epoch + 1:02d}/{epochs} | Loss: {loss.item():.4f}")


# ==============================================================================
# 5. MODEL EVALUATION
# ==============================================================================
# Disable gradient tracking for evaluation to save memory and compute
with torch.no_grad():
    # Get raw probability predictions for the test set
    test_predictions = model.forward(X_test_tensor)

    # Convert probabilities to binary classes using a 0.5 threshold
    predicted_classes = (test_predictions >= 0.5).float()

    # Calculate accuracy: compare predictions to true labels, convert booleans to float, and take mean
    accuracy = (predicted_classes == y_test_tensor).float().mean()

    print("\n" + "=" * 30)
    print(f"Final Test Accuracy: {accuracy.item() * 100:.2f}%")
