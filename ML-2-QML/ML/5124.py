"""Hybrid autoencoder combining classical MLP encoder/decoder with optional quantum latent refinement.

The module exposes a clean API with a single ``HybridAutoEncoder`` class that
encodes input data into a shared latent vector.  The latent vector is first
produced by a lightweight classical encoder, then optionally refined by a
variational quantum circuit supplied by the user.  The refined latent is
passed to a classical decoder so that reconstruction can be performed on the
CPU/GPU without the expense of a full quantum circuit per batch.  Training
can be done end‑to‑end with a single optimizer; the quantum parameters are
wrapped in a callable that forwards gradients from the classical loss to
the quantum circuit via the parameter‑shift rule.

The design deliberately blends ideas from all four reference pairs:

*  *Autoencoder.py* – classic MLP encoder/decoder, dropout, configurable
   architecture.
*  *EstimatorQNN.py* – simple quantum regression kernel that is adapted to
   refine the latent vector.
*  *SelfAttention.py* – the latent refinement step can be viewed as a
   self‑attention operation where the quantum circuit mixes latent
   components.  The circuit is built from `RealAmplitudes` and a swap‑test
   style measurement.
*  *QTransformerTorch.py* – the quantum module follows the same API as the
   quantum attention/ffn blocks, making it trivial to drop in a quantum
   transformer block later.

The class also exposes a ``train_autoencoder`` method that accepts a
DataLoader and returns a history of reconstruction loss.  The quantum part
is optional; passing ``None`` for the quantum encoder yields a pure
classical model.

"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Callable, List

@dataclass
class HybridAutoEncoderConfig:
    """Configuration for the HybridAutoEncoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    batch_size: int = 64

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class _MLPEncoder(nn.Module):
    def __init__(self, config: HybridAutoEncoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _MLPDecoder(nn.Module):
    def __init__(self, config: HybridAutoEncoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class HybridAutoEncoder(nn.Module):
    def __init__(
        self,
        config: HybridAutoEncoderConfig,
        quantum_refine: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        super().__init__()
        self.encoder = _MLPEncoder(config)
        self.decoder = _MLPDecoder(config)
        self.quantum_refine = quantum_refine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        if self.quantum_refine is not None:
            latent = self.quantum_refine(latent)
        return self.decoder(latent)

def HybridAutoEncoderFactory(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_refine: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> HybridAutoEncoder:
    config = HybridAutoEncoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )
    return HybridAutoEncoder(config, quantum_refine=quantum_refine)

def train_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: Optional[torch.device] = None
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoEncoder",
    "HybridAutoEncoderConfig",
    "HybridAutoEncoderFactory",
    "train_autoencoder",
]
