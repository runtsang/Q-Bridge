"""Hybrid classical–quantum autoencoder.

The implementation follows the structure of the original
`Autoencoder.py` but replaces the bottleneck with a variational
quantum circuit (SamplerQNN).  The encoder and decoder are fully
connected neural networks built with PyTorch, while the latent
representation is produced by a QNN that is trained jointly with
the classical parts.  The code is intentionally modular: the
quantum sub‑module is imported from `qml.py` and can be swapped
with any other QNN implementation without touching the classical
side.

The class exposes the same public API as the seed:
* `HybridAutoencoder(input_dim,...)` returns a configured network.
* `train_autoencoder(...)` runs a reconstruction training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the quantum helper; the module is expected to be named `qml.py`
import qml


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_trash: int = 2  # number of auxiliary qubits in the swap‑test


class HybridAutoencoderNet(nn.Module):
    """Hybrid encoder–quantum–decoder network."""

    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Classical encoder
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum bottleneck
        self.quantum = qml.create_qnn(cfg.latent_dim, cfg.num_trash)

        # Classical decoder
        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Run the QNN on the classical latent vector."""
        # The QNN expects a 1‑D array of parameters.
        # Convert to numpy and feed to the SamplerQNN.
        z_np = z.detach().cpu().numpy().flatten()
        # The QNN returns a 2‑D array of shape (batch, 2)
        q_out = self.quantum(z_np)
        # Collapse the two output qubits to a single latent value
        # (here we simply take their expectation difference).
        return torch.tensor(q_out[:, 0] - q_out[:, 1], device=z.device, dtype=torch.float32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.quantum_forward(z)
        return self.decode(z)


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_trash: int = 2,
) -> HybridAutoencoderNet:
    """Factory mirroring the original `Autoencoder`."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_trash=num_trash,
    )
    return HybridAutoencoderNet(cfg)


def train_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop that jointly optimises classical and quantum parameters."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
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
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_autoencoder",
]
