"""Hybrid classical autoencoder with kernel regularization and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Callable, Optional

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
# 1. RBF kernel – borrowed from the classical kernel module
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel suitable for latent regularisation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# 2. Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kernel_gamma: float = 1.0
    use_kernel_regularization: bool = False
    lambda_kernel: float = 1e-3

# --------------------------------------------------------------------------- #
# 3. Classical autoencoder network
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """Classical MLP encoder‑decoder with optional RBF kernel penalty."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Optional kernel module
        if config.use_kernel_regularization:
            self.kernel = RBFKernel(gamma=config.kernel_gamma)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def kernel_regularization(self, z: torch.Tensor) -> torch.Tensor:
        """Frobenius norm of (K – I) where K is the Gram matrix of latent vectors."""
        K = self.kernel(z, z)
        identity = torch.eye(K.size(0), device=K.device)
        diff = K - identity
        return torch.norm(diff, p='fro') ** 2

# --------------------------------------------------------------------------- #
# 4. Training routine
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train the autoencoder with optional kernel regularisation."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            if model.config.use_kernel_regularization:
                z = model.encode(batch)
                loss += model.config.lambda_kernel * model.kernel_regularization(z)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
# 5. Estimator wrapper (FastEstimator‑style)
# --------------------------------------------------------------------------- #
class HybridAutoencoderEstimator:
    """Estimator that evaluates the autoencoder on batches of inputs."""
    def __init__(self, model: HybridAutoencoderNet, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)

__all__ = [
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
    "HybridAutoencoderEstimator",
]
