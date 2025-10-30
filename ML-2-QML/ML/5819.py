"""Hybrid classical autoencoder with optional quantum decoder.

The class combines the fully‑connected autoencoder architecture from the
original Autoencoder.py with a quantum-inspired decoder that can be
replaced by a SamplerQNN or a simple feed‑forward network.  The
implementation keeps the same public API (``encode``, ``decode``,
``forward``) so that it can be used interchangeably in existing
pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert ``data`` to a float32 ``torch.Tensor`` on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# ---------------------------------------------------------------------------

@dataclass
class HybridAutoencoderConfig:
    """Hyper‑parameters for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_quantum_decoder: bool = False

# ---------------------------------------------------------------------------

class SamplerQNN(nn.Module):
    """A lightweight classical sampler network mirroring the QNN helper."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

# ---------------------------------------------------------------------------

class HybridAutoencoder(nn.Module):
    """A fully‑connected autoencoder that can optionally use a quantum decoder.

    The encoder is identical to the reference implementation but the
    decoder can be swapped with a quantum sampler.  The class exposes a
    ``set_quantum_decoder`` method to inject a callable that accepts a
    latent tensor and returns a reconstruction.
    """

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Classical decoder (default)
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Placeholder for a quantum decoder
        self.quantum_decoder: nn.Module | None = None
        if config.use_quantum_decoder:
            # A trivial fallback – users should call ``set_quantum_decoder``.
            self.quantum_decoder = SamplerQNN()

    # -----------------------------------------------------------------------

    def set_quantum_decoder(self, decoder: nn.Module) -> None:
        """Inject a quantum‑inspired decoder that accepts latent vectors."""
        self.quantum_decoder = decoder

    # -----------------------------------------------------------------------

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    # -----------------------------------------------------------------------

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.quantum_decoder is not None:
            return self.quantum_decoder(latents)
        return self.decoder(latents)

    # -----------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# ---------------------------------------------------------------------------

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_quantum_decoder: bool = False,
) -> HybridAutoencoder:
    """Convenience factory mirroring the original interface."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum_decoder=use_quantum_decoder,
    )
    return HybridAutoencoder(cfg)

# ---------------------------------------------------------------------------

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Standard reconstruction training loop.

    The loop is identical to the reference implementation but
    forwards through the hybrid decoder if one is present.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
