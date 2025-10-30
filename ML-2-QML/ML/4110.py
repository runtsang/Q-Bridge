"""Hybrid autoencoder combining classical convolutional encoder and quantum latent space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the quantum module; assumed to be in the same package
import QuantumAutoencoder as qml_module

def _as_tensor(data: torch.Tensor | torch.Tensor) -> torch.Tensor:
    """Ensure data is a torch float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder: classical conv encoder → quantum latent → classical decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder: a shallow CNN inspired by QCNN feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=config.dropout),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=config.dropout),
            nn.Flatten()
        )

        # Compute the flattened feature size after the encoder
        dummy_input = torch.zeros(1, 3, 32, 32)  # assuming 32x32 RGB images
        with torch.no_grad():
            feat_size = self.encoder(dummy_input).shape[1]

        # Quantum latent encoder
        self.quantum_latent = qml_module.HybridAutoencoder(
            input_dim=feat_size,
            latent_dim=config.latent_dim
        )

        # Classical decoder: simple MLP
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode images into latent vector using classical and quantum stages."""
        features = self.encoder(inputs)  # shape (B, feat_size)
        # Convert to numpy for the quantum circuit
        features_np = features.detach().cpu().numpy()
        # Quantum forward returns numpy array of shape (B, latent_dim)
        latent_np = self.quantum_latent.forward(features_np)
        # Convert back to torch tensor, preserving device and dtype
        latent = torch.tensor(latent_np, device=inputs.device, dtype=torch.float32)
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

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
    """Training loop for the hybrid autoencoder."""
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

__all__ = ["HybridAutoencoder", "AutoencoderConfig", "train_hybrid_autoencoder"]
