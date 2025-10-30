"""Hybrid classical autoencoder with optional quantum sampler layer.

The :class:`AutoencoderHybrid` implements a standard fully‑connected
autoencoder.  When ``config.use_quantum=True`` it inserts a
:class:`qiskit_machine_learning.neural_networks.SamplerQNN` as a
latent layer, allowing quantum‑generated embeddings to be learned.
The public interface (``forward``, ``train_autoencoder_hybrid``)
mirrors the original seed so that downstream code remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


@dataclass
class AutoencoderHybridConfig:
    """Configuration for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quantum: bool = False
    # Quantum options – only used when ``use_quantum=True``.
    quantum_circuit: Optional[object] = None
    quantum_weights: Optional[List[str]] = None
    sampler: Optional[Sampler] = None


class AutoencoderHybrid(nn.Module):
    """Hybrid classical / quantum autoencoder."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder.
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

        # Optional quantum sampler layer.
        self.quantum_sampler: Optional[SamplerQNN] = None
        if config.use_quantum:
            if config.quantum_circuit is None or config.quantum_weights is None:
                raise ValueError("Quantum circuit and weight names must be supplied.")
            self.quantum_sampler = SamplerQNN(
                circuit=config.quantum_circuit,
                input_params=[],
                weight_params=config.quantum_weights,
                sampler=config.sampler or Sampler(),
            )

        # Classical decoder.
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        if self.quantum_sampler is not None:
            # SamplerQNN expects a batch of weight vectors; we feed the latent directly.
            z = self.quantum_sampler(z)
        return self.decode(z)


def train_autoencoder_hybrid(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train a hybrid autoencoder and return the loss history."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


__all__ = ["AutoencoderHybridConfig", "AutoencoderHybrid", "train_autoencoder_hybrid"]
