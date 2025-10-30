"""Hybrid autoencoder combining a classical QCNN encoder with a quantum variational latent layer.

The class `HybridAutoencoderNet` implements a two-stage encoder: a classical
QCNN encoder followed by a quantum circuit that projects the feature vector
into a latent space.  The decoder is a fully‑connected network that
reconstructs the input from this latent representation.

Training is performed with a standard MSE loss; the quantum circuit is
treated as a black‑box function that is evaluated on the CPU.  The
quantum circuit can be updated separately using the `quantum_encode`
function provided in the QML module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Classical QCNN encoder
from.QCNN import QCNNModel

# Quantum encoder utilities
from.quantum_autoencoder import QuantumConvolutionAutoencoderQNN, quantum_encode


@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""

    input_dim: int
    latent_dim: int
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    qnn_input_dim: int = 8  # output of QCNNModel


class HybridAutoencoderNet(nn.Module):
    """Hybrid classical–quantum autoencoder.

    The encoder consists of a QCNN followed by a quantum variational circuit.
    The decoder is a standard fully‑connected network that maps the latent
    vector back to the input space.
    """

    def __init__(self, config: HybridAutoencoderConfig, qnn: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
        self.encoder = QCNNModel()

        # Quantum encoder
        if qnn is None:
            self.qnn = QuantumConvolutionAutoencoderQNN(
                input_dim=config.qnn_input_dim,
                latent_dim=config.latent_dim,
            )
        else:
            self.qnn = qnn

        # Decoder
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode, quantum‑encode, and decode."""
        # Classical feature extraction
        features = self.encoder(inputs)  # shape: (batch, 8)

        # Quantum latent vector
        # Convert to numpy for the quantum backend
        features_np = features.detach().cpu().numpy()
        latent_np = quantum_encode(self.qnn, features_np)  # shape: (batch, latent_dim)
        latent = torch.from_numpy(latent_np).to(inputs.device)

        # Reconstruction
        return self.decoder(latent)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid autoencoder using a standard MSE loss.

    Parameters
    ----------
    model : HybridAutoencoderNet
        The hybrid network to train.
    data : torch.Tensor
        Training data of shape (N, input_dim).
    epochs : int, default 100
        Number of training epochs.
    batch_size : int, default 64
        Batch size.
    lr : float, default 1e-3
        Learning rate.
    weight_decay : float, default 0.0
        Weight decay for Adam.
    device : torch.device | None, default None
        Device to train on; defaults to CUDA if available.

    Returns
    -------
    list[float]
        History of training loss per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1:03d}/{epochs:03d}  loss={epoch_loss:.6f}")

    return history


__all__ = ["HybridAutoencoderNet", "HybridAutoencoderConfig", "train_hybrid_autoencoder"]
