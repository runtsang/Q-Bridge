"""Extended classical regression model with Lightning integration and richer architecture."""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data from a superposition-inspired function.
    The function is sin(∑x_i) + 0.1*cos(2∑x_i) with optional Gaussian noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """Simple residual block used in the regression network."""
    def __init__(self, size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class QModel(nn.Module):
    """Classical feed‑forward regression model with residual connections."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

class LightningModel(pl.LightningModule):
    """PyTorch Lightning wrapper that exposes training logic."""
    def __init__(self, num_features: int, lr: float = 1e-3):
        super().__init__()
        self.model = QModel(num_features)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds = self(batch["states"])
        loss = nn.functional.mse_loss(preds, batch["target"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["states"])
        loss = nn.functional.mse_loss(preds, batch["target"])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "LightningModel"]
