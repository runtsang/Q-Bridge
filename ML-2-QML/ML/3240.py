"""Hybrid autoencoder + classifier implementation for classical ML.

This module provides a single class, `HybridAutoEncoderClassifier`, that encapsulates a
fully‑connected auto‑encoder followed by a feed‑forward classifier.  The auto‑encoder
acts as a feature extractor producing a latent representation that is fed into the
classifier.  The implementation builds on the patterns from the original
`QuantumClassifierModel` and `Autoencoder` seeds but combines them into a coherent
end‑to‑end pipeline.

Key features:
* `AutoencoderNet` – a lightweight MLP auto‑encoder with configurable depth.
* `ClassifierNet` – a multi‑layer perceptron that accepts the latent vector.
* `HybridAutoEncoderClassifier` – exposes `train_autoencoder`, `train_classifier`,
  `fit`, `predict`, and `latent` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Auto‑encoder block
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Encoder–decoder network with symmetric hidden layers."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Factory that returns a configured auto‑encoder."""
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Classifier block
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier that mirrors the quantum helper interface.
    Returns the network, a list of input feature indices, weight sizes and output
    observables (here trivial indices).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
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
    # For compatibility with the quantum side we expose simple metadata
    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# Hybrid wrapper
# --------------------------------------------------------------------------- #

class HybridAutoEncoderClassifier(nn.Module):
    """
    End‑to‑end pipeline that first reduces dimensionality with an auto‑encoder
    and then classifies the latent representation.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        ae_cfg = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = Autoencoder(ae_cfg)
        self.classifier, self.enc_idx, self.w_sizes, self.obs = build_classifier_circuit(
            num_features=latent_dim,
            depth=classifier_depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.classifier(z)

    # --------------------------------------------------------------------- #
    # Training helpers
    # --------------------------------------------------------------------- #

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
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
            for (batch,) in loader:
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

    def train_classifier(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        *,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)
        dataset = TensorDataset(
            self._as_tensor(self.autoencoder.encode(data).detach()),
            self._as_tensor(labels),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch_x, batch_y) in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.classifier(batch_x)
                loss = loss_fn(logits, batch_y.long())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def fit(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        *,
        ae_epochs: int = 50,
        clf_epochs: int = 30,
        **kwargs,
    ) -> None:
        """Convenience wrapper that trains the auto‑encoder first, then the classifier."""
        self.train_autoencoder(data, epochs=ae_epochs, **kwargs)
        self.train_classifier(data, labels, epochs=clf_epochs, **kwargs)

    def predict(self, data: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        with torch.no_grad():
            logits = self.forward(self._as_tensor(data).to(device))
        return torch.argmax(logits, dim=-1)

    def latent(self, data: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return the latent representation produced by the auto‑encoder."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.eval()
        with torch.no_grad():
            return self.autoencoder.encode(self._as_tensor(data).to(device))

    @staticmethod
    def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        return torch.as_tensor(data, dtype=torch.float32)

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "build_classifier_circuit",
    "HybridAutoEncoderClassifier",
]
