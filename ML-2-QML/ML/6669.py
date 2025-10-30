import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class QCNNModel(nn.Module):
    """
    A lightweight classical front‑end that mimics the convolutional stages of the
    quantum circuit but with trainable linear layers.  It can be dropped into any
    PyTorch pipeline that expects a 1‑D feature tensor of length 8.
    """
    def __init__(self, in_features: int = 8, hidden: int = 16, out_features: int = 1):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QCNNEnhanced(QCNNModel):
    """Enhanced QCNN model with optional deeper hidden layers."""
    def __init__(self, in_features: int = 8, hidden: int = 32, out_features: int = 1):
        super().__init__(in_features, hidden, out_features)

def QCNN() -> QCNNModel:
    """Return a freshly initialised QCNNModel."""
    return QCNNModel()

def train_classical(dataset: TensorDataset,
                    model: nn.Module,
                    epochs: int = 10,
                    lr: float = 1e-3,
                    batch_size: int = 32) -> None:
    """
    Simple training routine for the classical QCNNModel.  The routine is
    intentionally lightweight – it is designed to be used as a reference
    implementation or as part of a larger hybrid workflow.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb.float())
            loss = criterion(preds.squeeze(), yb.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(loader):.4f}")

__all__ = ["QCNN", "QCNNModel", "QCNNEnhanced", "train_classical"]
