"""Hybrid classical‑quantum autoencoder combining classical and quantum components.

The module defines:
* QFCModel – classical convolutional feature extractor (Quantum‑NAT style).
* SamplerQNN – lightweight neural network used as a quantum latent layer.
* AutoencoderConfig, AutoencoderNet – classical autoencoder backbone.
* HybridAutoencoder – integrates QFCModel, AutoencoderNet, and a quantum latent
  layer (callable or nn.Module). The quantum layer is expected to map a
  latent vector to a probability distribution; by default it uses a wrapper
  that calls the quantum module from the accompanying QML package.
* train_autoencoder – training loop for the classical parts; the quantum layer
  is treated as a black box during training.

The design allows swapping the quantum backend without changing the
classical architecture, demonstrating a true hybrid scaling paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable, Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- Classical feature extractor -----------------------------------------

class QFCModel(nn.Module):
    """Classical CNN + fully‑connected projection (Quantum‑NAT style)."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --- Classical sampler network --------------------------------------------

class SamplerQNN(nn.Module):
    """Simple neural network that mimics a quantum sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(inputs), dim=-1)

# --- Classical autoencoder backbone ---------------------------------------

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected encoder‑decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    return AutoencoderNet(
        AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    )

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --- Hybrid autoencoder --------------------------------------------------

class HybridAutoencoder(nn.Module):
    """
    Combines the classical feature extractor, encoder/decoder, and a quantum
    latent layer. The quantum layer is passed as a callable that accepts a
    latent tensor and returns a probability vector of the same batch size.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        quantum_layer: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        *,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_extractor = QFCModel()
        self.encoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.quantum_layer = quantum_layer
        # Decoder expects the same dimension as the quantum output
        self.decoder = Autoencoder(
            input_dim=latent_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        latent = self.encoder.encode(features)
        quantum_out = self.quantum_layer(latent)
        # Pass the quantum measurement distribution through the decoder
        return self.decoder.decode(quantum_out)

__all__ = [
    "QFCModel",
    "SamplerQNN",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "HybridAutoencoder",
]
