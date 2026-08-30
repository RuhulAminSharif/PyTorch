from sklearn.datasets import make_classification
import torch
from torch.utils.data import Dataset, DataLoader

# Step 1: Create a synthetic classification dataset using sklearn
X, y = make_classification(
    n_samples=10,  # Number of samples
    n_features=2,  # Number of features
    n_informative=2,  # Number of informative features
    n_redundant=0,  # Number of redundant features
    n_classes=2,  # Number of classes
    random_state=42,  # For reproducibility
)
# Convert the data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


class CustomDataset(Dataset):
    def __init__(self, features, labels) -> None:
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


dataset = CustomDataset(X, y)

print(len(dataset))
print(dataset[0])
print(dataset[2])

dataloader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    # collate_fn
    # sampler
)
for batch_X, batch_y in dataloader:
    print(batch_X)
    print(batch_y)
    print("*"*50)
