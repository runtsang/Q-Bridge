"""Enhanced autoencoder with fast estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable, Sequence

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Helper
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(data, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

# --------------------------------------------------------------------------- #
# Configuration
@dataclass
class AutoencoderGen132Config:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    device: torch.device | None = None

# --------------------------------------------------------------------------- #
# Core Network
class AutoencoderGen132Net(nn.Module):
    """Encoder–decoder network with configurable depth and dropout."""
    def __init__(self, config: AutoencoderGen132Config) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            out_dim=config.input_dim,
            dropout=config.dropout,
        )
        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z.to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Fast Estimator Utilities
class FastBaseEstimator:
    """Evaluates a model on batches of inputs, returning scalar observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32, device=self.device)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None) -> None:
        super().__init__(model)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Training Routine
def train_autoencoder(
    model: AutoencoderGen132Net,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> List[float]:
    """Standard MSE reconstruction training loop."""
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(model.device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Public API
def AutoencoderGen132(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderGen132Net:
    """Factory constructing a fully‑connected autoencoder."""
    cfg = AutoencoderGen132Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderGen132Net(cfg)

__all__ = [
    "AutoencoderGen132",
    "AutoencoderGen132Net",
    "AutoencoderGen132Config",
    "train_autoencoder",
    "FastBaseEstimator",
    "FastEstimator",
]
