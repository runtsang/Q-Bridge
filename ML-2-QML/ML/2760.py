from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Optional import of the quantum encoder – the module is expected to live
# in the same package.  If it is missing, a dummy identity encoder is
# provided so that the rest of the code remains importable.
try:
    from.quantum_encoder import QuantumEncoder
except Exception:  # pragma: no cover
    class QuantumEncoder(nn.Module):
        def __init__(self, *_, **__):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x  # identity fallback


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
class AutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum parameters
    qreg_size: int | None = None
    qlayer_reps: int = 5

    def __post_init__(self) -> None:
        # If the quantum register size is not specified, default to the
        # classical latent dimension so that the quantum encoder can
        # return a vector of the same size.
        if self.qreg_size is None:
            self.qreg_size = self.latent_dim


class ClassicalEncoder(nn.Module):
    """MLP encoder that maps input vectors to a latent representation."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalDecoder(nn.Module):
    """MLP decoder that reconstructs the input from a latent vector."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridAutoencoder(nn.Module):
    """Hybrid auto‑encoder that uses a quantum encoder for the latent
    representation.  The quantum encoder transforms the classical
    latent vector into a quantum‑derived representation which is then
    decoded classically.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ClassicalEncoder(cfg)
        self.quantum_encoder = QuantumEncoder(
            latent_dim=cfg.latent_dim,
            qreg_size=cfg.qreg_size,
            reps=cfg.qlayer_reps,
        )
        # The decoder expects the output of the quantum encoder, which
        # has dimensionality equal to the quantum register size.
        self.decoder = ClassicalDecoder(
            AutoencoderConfig(
                input_dim=cfg.input_dim,
                latent_dim=cfg.qreg_size,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
                qreg_size=cfg.qreg_size,
                qlayer_reps=cfg.qlayer_reps,
            )
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def quantum_encode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.quantum_encoder(latent)

    def decode(self, q_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(q_latent)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        q_latent = self.quantum_encode(latent)
        recon = self.decode(q_latent)
        return recon


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    qreg_size: int | None = None,
    qlayer_reps: int = 5,
) -> HybridAutoencoder:
    """Convenience factory that mirrors the classical Autoencoder factory."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        qreg_size=qreg_size,
        qlayer_reps=qlayer_reps,
    )
    return HybridAutoencoder(cfg)


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
    """Simple reconstruction training loop for the hybrid model."""
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
    "AutoencoderConfig",
    "ClassicalEncoder",
    "ClassicalDecoder",
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
