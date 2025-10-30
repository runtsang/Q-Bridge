import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Iterable, List

class FCL(nn.Module):
    """
    Multi‑layer fully connected network with optional dropout and batch‑norm.
    Provides a complete training pipeline and an inference routine that
    accepts a list of parameters (thetas) similar to the original seed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self._act(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def _act(name: str) -> nn.Module:
        if name.lower() == "relu":
            return nn.ReLU()
        if name.lower() == "tanh":
            return nn.Tanh()
        if name.lower() == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original seed interface: accept a list of parameters
        and return the network output as a NumPy array.
        """
        with torch.no_grad():
            # Load parameters into the model
            flat_params = torch.tensor(list(thetas), dtype=torch.float32)
            idx = 0
            for param in self.parameters():
                numel = param.numel()
                param.data.copy_(flat_params[idx : idx + numel].view_as(param))
                idx += numel
            # Dummy input: zeros of appropriate shape
            dummy = torch.zeros((1, self.network[0].in_features))
            out = self.forward(dummy)
            return out.detach().cpu().numpy()

    def train_network(
        self,
        train_loader: DataLoader,
        epochs: int,
        lr: float = 1e-3,
        loss_fn: nn.Module | None = None,
        device: str | None = None,
    ) -> None:
        """
        Train the network on a PyTorch DataLoader.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = loss_fn or nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = self(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            epoch_loss /= len(train_loader.dataset)
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    def evaluate(
        self,
        data_loader: DataLoader,
        loss_fn: nn.Module | None = None,
        device: str | None = None,
    ) -> tuple[float, float]:
        """
        Return average loss and mean absolute error on a dataset.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        loss_fn = loss_fn or nn.MSELoss()
        self.eval()
        total_loss = 0.0
        total_mae = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                pred = self(x)
                total_loss += loss_fn(pred, y).item() * x.size(0)
                total_mae += F.l1_loss(pred, y, reduction="sum").item()
        n = len(data_loader.dataset)
        return total_loss / n, total_mae / n

    def get_flat_params(self) -> np.ndarray:
        """Return all parameters flattened into a 1‑D NumPy array."""
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params: Iterable[float]) -> None:
        """Set all parameters from a flattened array."""
        flat_params = np.asarray(flat_params, dtype=np.float32)
        idx = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(torch.from_numpy(flat_params[idx : idx + numel]).view_as(param))
            idx += numel

    def save_weights(self, path: str) -> None:
        """Persist the model weights to disk."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location="cpu"))

__all__ = ["FCL"]
