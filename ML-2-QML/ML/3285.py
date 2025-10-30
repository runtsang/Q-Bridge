"""Hybrid classical autoencoder + classifier for feature extraction and classification.

Builds a classical autoencoder to compress high‑dimensional input data into a latent
space, then feeds the latent representation into a small feed‑forward classifier.
The two components are designed to be interchangeable with their quantum counterparts
in the QML module, enabling seamless hybrid experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.as_tensor(data, dtype=torch.float32)
    if t.dtype!= torch.float32:
        t = t.to(dtype=torch.float32)
    return t


@dataclass
class HybridConfig:
    """Configuration for the hybrid model."""
    input_dim: int
    latent_dim: int = 32
    encoder_hidden: Tuple[int,...] = (128, 64)
    decoder_hidden: Tuple[int,...] = (64, 128)
    classifier_hidden: Tuple[int,...] = (64,)
    dropout: float = 0.1
    batch_size: int = 64
    epochs_auto: int = 100
    epochs_cls: int = 50
    lr_auto: float = 1e-3
    lr_cls: float = 1e-3
    weight_decay: float = 0.0


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.encoder_hidden:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.decoder_hidden):
            decoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ClassifierNet(nn.Module):
    """Classifier that takes latent vectors."""
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for h in config.classifier_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HybridClassifierAutoencoder(nn.Module):
    """Combines an autoencoder and a classifier into a single module."""
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config
        self.autoencoder = AutoencoderNet(config)
        self.classifier = ClassifierNet(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def train_autoencoder(self, data: torch.Tensor) -> list[float]:
        """Train only the autoencoder, keeping the classifier fixed."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.autoencoder.parameters(),
                               lr=self.config.lr_auto,
                               weight_decay=self.config.weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []
        for _ in range(self.config.epochs_auto):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                recon = self.autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def train_classifier(self,
                         data: torch.Tensor,
                         labels: torch.Tensor) -> list[float]:
        """Train the classifier on the encoded latent vectors."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.autoencoder.eval()
        dataset = TensorDataset(_as_tensor(data), _as_tensor(labels))
        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True)
        opt = torch.optim.Adam(self.classifier.parameters(),
                               lr=self.config.lr_cls,
                               weight_decay=self.config.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        history: list[float] = []
        for _ in range(self.config.epochs_cls):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                z = self.autoencoder.encode(x_batch)
                logits = self.classifier(z)
                loss = loss_fn(logits, y_batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * x_batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def accuracy(self,
                 data: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Return classification accuracy on the provided data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        with torch.no_grad():
            z = self.autoencoder.encode(_as_tensor(data).to(device))
            logits = self.classifier(z)
            preds = logits.argmax(dim=1)
            acc = (preds == _as_tensor(labels).to(device)).float().mean().item()
        return acc


__all__ = [
    "HybridConfig",
    "AutoencoderNet",
    "ClassifierNet",
    "HybridClassifierAutoencoder",
]
