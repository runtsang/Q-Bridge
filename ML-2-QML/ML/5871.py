from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, List, Optional

from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Classical regression network
# --------------------------------------------------------------------------- #
class Regressor(nn.Module):
    """Tiny two‑layer feed‑forward regressor."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
#  Auto‑encoder configuration and model
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Auto‑encoder built from a configuration."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
#  Unified hybrid model
# --------------------------------------------------------------------------- #
class UnifiedEstimatorAutoencoder:
    """
    Combines a regression network, an auto‑encoder, and an optional quantum
    evaluator.  The quantum evaluator must be supplied from the QML module
    and should accept a latent vector and return a scalar torch.Tensor.
    """

    def __init__(
        self,
        regressor: Optional[nn.Module] = None,
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
        quantum_evaluator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.regressor = regressor or Regressor()
        self.autoencoder_cfg = autoencoder_cfg or AutoencoderConfig(input_dim=2)
        self.autoencoder = AutoencoderNet(self.autoencoder_cfg)
        self.quantum_evaluator = quantum_evaluator

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.float()
        return torch.as_tensor(data, dtype=torch.float32)

    # ------------------------------------------------------------------
    #  Training utilities
    # ------------------------------------------------------------------
    def train_autoencoder(
        self,
        data: Iterable[float] | torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        dataset = TensorDataset(self._as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def train_regressor(
        self,
        X: Iterable[float] | torch.Tensor,
        y: Iterable[float] | torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regressor.to(device)
        X = self._as_tensor(X)
        y = self._as_tensor(y).unsqueeze(-1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.regressor(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    # ------------------------------------------------------------------
    #  Quantum interface
    # ------------------------------------------------------------------
    def evaluate_quantum(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward the latent vector through the quantum evaluator.
        If no evaluator is supplied, returns a zero tensor of matching shape.
        """
        if self.quantum_evaluator is None:
            return torch.zeros_like(latent.sum(dim=-1, keepdim=True))
        return self.quantum_evaluator(latent)

__all__ = [
    "Regressor",
    "AutoencoderConfig",
    "AutoencoderNet",
    "UnifiedEstimatorAutoencoder",
]
