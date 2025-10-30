"""
HybridAutoencoder: Classical encoder + Quantum decoder.

This module implements a PyTorch autoencoder where the encoder is a
multi‑layer perceptron and the decoder is a variational quantum circuit.
The quantum decoder is defined in the accompanying QML module and returns
expectation values that are interpreted as the reconstructed features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn

# Import the quantum decoder defined in the QML module.
# The QML module must be available in the same package.
try:
    from.quantum_decoder import QuantumDecoder
except Exception:  # pragma: no cover
    # In environments where the QML module is not yet available,
    # provide a stub that raises an informative error.
    class QuantumDecoder:  # type: ignore
        def __init__(self, *_, **__):
            raise RuntimeError("QuantumDecoder module not found. "
                               "Ensure the QML module is installed.")

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    """A PyTorch autoencoder with a quantum decoder."""
    def __init__(self, config: AutoencoderConfig, q_decoder: QuantumDecoder | None = None) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
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

        # Quantum decoder
        self.q_decoder = q_decoder or QuantumDecoder(config.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input into latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent vector using the quantum decoder.
        The quantum decoder returns a NumPy array; convert it back to a torch tensor.
        """
        # Move to CPU for simulation
        z_np = z.detach().cpu().numpy()
        # Decode each sample in the batch
        decoded = []
        for latent in z_np:
            out = self.q_decoder.run(latent)
            decoded.append(out)
        decoded = np.stack(decoded, axis=0)
        return torch.from_numpy(decoded).to(z.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        z = self.encode(x)
        return self.decode(z)

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
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

__all__ = ["AutoencoderConfig", "HybridAutoencoder", "train_hybrid_autoencoder"]
