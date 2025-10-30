"""Hybrid classical autoencoder + classifier.

Combines a lightweight fully‑connected autoencoder with a feed‑forward
classifier that operates on the latent representation.  The design
mirrors the quantum helper interface so that the same factory functions
can be swapped in the quantum implementation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# Autoencoder utilities
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Encoder‑decoder MLP with configurable hidden layers."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(data, dtype=torch.float32)
    return t if t.dtype == torch.float32 else t.to(dtype=torch.float32)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Hybrid classifier
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a feed‑forward classifier and metadata for the quantum API."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

class HybridAutoencoderClassifier(nn.Module):
    """Classical pipeline: autoencoder → latent embedding → classifier."""
    def __init__(
        self,
        input_dim: int,
        *,
        ae_latent: int = 32,
        ae_hidden: Tuple[int, int] = (128, 64),
        ae_dropout: float = 0.1,
        clf_depth: int = 3,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=ae_latent,
            hidden_dims=ae_hidden,
            dropout=ae_dropout,
        )
        self.classifier, _, _, _ = build_classifier_circuit(ae_latent, clf_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        return logits

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "build_classifier_circuit",
    "HybridAutoencoderClassifier",
]
