"""Hybrid classical autoencoder with optional classifier head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class AutoencoderGen089Config:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    classifier_depth: int = 2
    use_classifier: bool = False

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Coerce data to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

class AutoencoderGen089(nn.Module):
    """A fully‑connected autoencoder optionally augmented with a classifier head."""
    def __init__(self, cfg: AutoencoderGen089Config) -> None:
        super().__init__()
        self.cfg = cfg

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

        # Optional classifier head
        self.classifier: Optional[nn.Module] = None
        if cfg.use_classifier:
            cls_layers: List[nn.Module] = []
            in_dim = cfg.latent_dim
            for _ in range(cfg.classifier_depth):
                cls_layers.append(nn.Linear(in_dim, cfg.latent_dim))
                cls_layers.append(nn.ReLU())
                in_dim = cfg.latent_dim
            cls_layers.append(nn.Linear(in_dim, 2))
            self.classifier = nn.Sequential(*cls_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        recon = self.decode(z)
        if self.classifier:
            cls = self.classifier(z)
            return recon, cls
        return recon

def build_autoencoder_gen089(cfg: AutoencoderGen089Config) -> AutoencoderGen089:
    """Factory mirroring the quantum helper."""
    return AutoencoderGen089(cfg)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier mirroring the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

def train_autoencoder_gen089(
    model: AutoencoderGen089,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    loss_fn: Callable = nn.MSELoss(),
    classification_loss_fn: Callable = nn.CrossEntropyLoss(),
    labels: Optional[torch.Tensor] = None,
) -> List[float]:
    """Simple reconstruction training loop with optional classification."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if labels is None:
        labels = torch.zeros((data.size(0),), dtype=torch.long)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, lbl in loader:
            batch = batch.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if model.classifier:
                recon, cls = output
                loss = loss_fn(recon, batch) + classification_loss_fn(cls, lbl)
            else:
                loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderGen089",
    "AutoencoderGen089Config",
    "build_autoencoder_gen089",
    "train_autoencoder_gen089",
    "build_classifier_circuit",
]
