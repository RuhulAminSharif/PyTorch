import os
import json
import random
import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 0. CONFIG
# ==============================================================================
@dataclass
class Config:
    data_url: str = (
        "https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv"
    )
    target_col: str = "diagnosis"
    id_cols: tuple = ("id", "Unnamed: 32")

    test_size: float = 0.2
    val_size: float = 0.2
    random_seed: int = 42

    batch_size: int = 32
    hidden_dims: tuple = (
        32,
        16,
    )
    dropout: float = 0.2

    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    epochs: int = 10000
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.001

    use_class_weighted_sampler: bool = True

    output_dir: str = "artifacts"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


# ==============================================================================
# 1. REPRODUCIBILITY
# ==============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.random_seed)

# A dedicated generator for DataLoader shuffling so it's reproducible too.
g = torch.Generator()
g.manual_seed(cfg.random_seed)

# ==============================================================================
# 2. HARDWARE CONFIGURATION
# ==============================================================================
# Automatically select GPU (CUDA/MPS) if available, otherwise fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")


# ==============================================================================
# 3. DATA LOADING & PREPARATION
# ==============================================================================
def load_data(cfg: Config) -> pd.DataFrame:
    try:
        df = pd.read_csv(cfg.data_url)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {cfg.data_url}: {e}")

    drop_cols = [c for c in cfg.id_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


df = load_data(cfg=cfg)


# Separate features (X) and target labels (y)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Split data into training, validation and test sets

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=np.asarray(y)
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=cfg.val_size,
    random_state=cfg.random_seed,
    stratify=np.asarray(y_train_full),
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================================================================
# 4. DATA PREPROCESSING
# ==============================================================================
# Standardize features (fit on train, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Encode categorical labels ('M'/'B') to numerical (1/0)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)
y_test_encoded = encoder.transform(y_test)

joblib.dump(scaler, os.path.join(cfg.output_dir, "scaler.joblib"))
joblib.dump(encoder, os.path.join(cfg.output_dir, "label_encoder.joblib"))

# Convert NumPy arrays to PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train_scaled).to(torch.float32)
X_val_tensor = torch.from_numpy(X_val_scaled).to(torch.float32)
X_test_tensor = torch.from_numpy(X_test_scaled).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train_encoded).view(-1, 1).to(torch.float32)
y_val_tensor = torch.from_numpy(y_val_encoded).view(-1, 1).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test_encoded).view(-1, 1).to(torch.float32)


# Create Datasets (pairs X and y together)
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)


# --- CUSTOM COLLATOR ---
def custom_collate_fn(batch):
    """
    Receives a list of tuples from the Dataset: [(feature_0, label_0), ...]
    Stacks them into batched tensors.
    we can also apply dynamic padding here for NLP/Time-Series.
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Stack lists of tensors into a single tensor block [batch_size, num_features]
    features = torch.stack(features)
    labels = torch.stack(labels)

    return features, labels


"""
# pin_memory only helps (and is only valid) on CUDA; on CPU/MPS
# it's wasted or can even error out on some setups.
use_pin_memory = device.type == "cuda"
# IMPROVEMENT: num_workers > 0 can hang/error on Windows or in notebook
# environments unless guarded by `if __name__ == "__main__"`. We set it
# conditionally and wrap execution below.
num_workers = 2 if os.name != "nt" else 0
"""
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,  # MUST be False when using a custom sampler
    num_workers=2 if os.name != "nt" else 0,
    pin_memory=device.type == "cuda",
    drop_last=True,  # Drops the last incomplete batch for batchnorm stability
    collate_fn=custom_collate_fn,
    generator=g,
    # sampler
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=2 if os.name != "nt" else 0,
    pin_memory=device.type == "cuda",
    drop_last=False,
    collate_fn=custom_collate_fn,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,  # Never shuffle validation/test sets
    num_workers=2 if os.name != "nt" else 0,
    pin_memory=device.type == "cuda",
    drop_last=False,  # Evaluate all data, even incomplete final batches
    collate_fn=custom_collate_fn,
)


# ==============================================================================
# 5. MODEL DEFINITION
# ==============================================================================
class SimpleNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Defining layers
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.network(features)


# ==============================================================================
# 6. EARLY STOPPING HELPER
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ==============================================================================
# 7. TRAINING PIPELINE (With Validation & Checkpointing)
# ==============================================================================


def train(cfg: Config) -> tuple[nn.Module, str]:
    # Instantiate model and move it to the configured device
    model = SimpleNN(X_train_tensor.shape[1]).to(device)

    # Loss function and Optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Tracking best validation loss for model checkpointing
    early_stopper = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_val_loss = float("inf")
    model_save_path = os.path.join(cfg.output_dir, "best_model.pth")

    history = []

    logger.info("Starting training loop...")

    for epoch in range(cfg.epochs):

        # --- TRAINING ---
        model.train()  # Set model to training mode
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            # Move batch data to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 1. Forward Pass
            y_pred = model(batch_X)

            # 2. Loss Calculation
            loss = loss_function(y_pred, batch_y)

            # 3. Clear existing gradients
            optimizer.zero_grad()

            # 4. Backward Pass (Calculate gradients)
            loss.backward()

            # 5. Update Weights
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item() * batch_X.size(0)

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader.dataset)  # type: ignore
        # avg_train_loss = train_loss / (len(train_loader) * batch_size) # use this if drop_last = True

        # --- VALIDATION LOOP ---
        avg_val_loss, val_accuracy, _ = evaluate(model, val_loader, loss_function)

        scheduler.step(avg_val_loss)

        # --- MODEL CHECKPOINTING ---
        checkpoint_msg = ""

        # Save the model weights if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": asdict(cfg),
                },
                model_save_path,
            )
            checkpoint_msg = " -> Model Saved!"
        else:
            checkpoint_msg = ""

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        # Log progress every 5 epochs
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(
                f"Epoch: {epoch + 1:03d}/{cfg.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy*100:.2f}%{checkpoint_msg}"
            )

        # --- EARLY STOPPING CHECK ---
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
            break
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, model_save_path


# ==============================================================================
# 8. EVALUATION
# ==============================================================================
def evaluate(model, loader, loss_function):
    model.eval()  # Set model to evaluation mode (disables dropout/batchnorm updates)
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():  # Disable gradient tracking to save memory/compute
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            pred = model(batch_X)
            loss = loss_function(pred, batch_y)
            total_loss += loss.item() * batch_X.size(0)

            # Calculate accuracy
            predicted_classes = (pred >= 0.5).float()
            all_preds.append(predicted_classes.cpu())
            all_targets.append(batch_y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy, (all_targets, all_preds)


if __name__ == "__main__":
    model, model_save_path = train(cfg)

    logger.info("=" * 40)
    logger.info("Evaluating Best Model on Test Data...")

    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded weights from epoch {checkpoint['epoch'] + 1}")

    loss_function = nn.BCELoss()
    test_loss, test_accuracy, (targets, preds) = evaluate(
        model, test_loader, loss_function
    )

    # IMPROVEMENT: report precision/recall/F1/confusion matrix, not just
    # accuracy — accuracy alone is misleading on imbalanced medical data like
    # this breast-cancer dataset.
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    cm = confusion_matrix(targets, preds)

    logger.info(f"Final Test Loss:      {test_loss:.4f}")
    logger.info(f"Final Test Accuracy:  {test_accuracy * 100:.2f}%")
    logger.info(f"Final Test Precision: {precision:.4f}")
    logger.info(f"Final Test Recall:    {recall:.4f}")
    logger.info(f"Final Test F1:        {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )
