"""Classical regression dataset and Lightning model extending the original."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data in the form of a superposition of |0> and |1> states
    projected onto a real‑valued target.  An optional Gaussian noise can
    be added to the target to simulate measurement error.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(0, noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset that yields state tensors and target scalars."""

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualNet(nn.Module):
    """
    A small residual network that maps input features to a single output.
    The residual block improves gradient flow and allows the model to
    express identity mappings more easily.
    """

    def __init__(self, num_features: int, hidden: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(num_features, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out += residual  # residual connection
        out = self.layer3(out)
        return out.squeeze(-1)


class RegressionLitModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper that trains the ResidualNet on the
    RegressionDataset.  Early stopping and learning‑rate scheduling
    are enabled by default.
    """

    def __init__(self, num_features: int, lr: float = 1e-3):
        super().__init__()
        self.model = ResidualNet(num_features)
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds = self(batch["states"])
        loss = self.criterion(preds, batch["target"])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["states"])
        loss = self.criterion(preds, batch["target"])
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=5, factor=0.5
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


__all__ = ["RegressionDataset", "RegressionLitModel", "generate_superposition_data", "ResidualNet"]
