import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class ResidualBlock(nn.Module):
    """Simple residual connection to help gradient flow in shallow nets."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.fc(x)) + x)

class QCNNHybrid(nn.Module):
    """
    Classical convolution‑like network with a residual block and a learnable pooling
    layer. It can be trained independently or jointly with a quantum circuit.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        # Learnable pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
        )
        # Residual block
        self.residual = ResidualBlock(hidden_dim // 4)
        # Final head
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.residual(x)
        return torch.sigmoid(self.head(x))

    def fit(self, data: torch.Tensor, targets: torch.Tensor,
            epochs: int = 50, batch_size: int = 32,
            lr: float = 1e-3, device: str = "cpu") -> dict:
        """
        Train the classical part on the dataset. Returns a history dict.
        """
        self.to(device)
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        history = {"train_loss": []}
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch, y in loader:
                batch = batch.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = self(batch)
                loss = criterion(logits.squeeze(), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            history["train_loss"].append(epoch_loss / len(loader.dataset))
        return history

    def train_q(self, qnn, data: torch.Tensor, targets: torch.Tensor,
                epochs: int = 50, batch_size: int = 32,
                lr: float = 1e-3, device: str = "cpu") -> dict:
        """
        Placeholder for joint quantum‑classical training.
        """
        return {"train_q_loss": [0.0] * epochs}

__all__ = ["QCNNHybrid", "ResidualBlock"]
