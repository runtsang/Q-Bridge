"""Hybrid quantum‑classical autoencoder using a quanvolution filter.

The quantum branch applies a random two‑qubit quantum kernel to each 2×2
image patch, producing a 4‑dimensional feature vector per patch.  The
classical decoder reconstructs the 28×28 image from the concatenated
features.  The implementation uses TorchQuantum for the quantum
operations and standard PyTorch layers for the decoder.

The module supports a simple `encode`/`decode` interface and a training
loop similar to the classical version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """
    Apply a random two‑qubit quantum kernel to 2×2 image patches.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Concatenated feature vector of shape (B, 784).
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (B, 784)


# --------------------------------------------------------------------------- #
# Quantum‑classical autoencoder
# --------------------------------------------------------------------------- #
class QuanvolutionAutoencoder(tq.QuantumModule):
    """
    Hybrid autoencoder that uses a quantum quanvolution filter as the
    encoder and a fully‑connected decoder to reconstruct the image.
    """
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Decoder: fully‑connected network mapping 784 → latent_dim → 784
        decoder_layers = []
        in_dim = 784
        for hidden in hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, latent_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(latent_dim, 784))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (B, 1, 28, 28).
        """
        features = self.qfilter(x)  # (B, 784)
        recon_flat = self.decoder(features)  # (B, 784)
        recon = recon_flat.view(x.size(0), 1, 28, 28)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        features = self.qfilter(x)
        latent = self.decoder[0](features)  # first linear layer
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to image space."""
        recon_flat = self.decoder[1:](z)
        return recon_flat.view(-1, 1, 28, 28)


# --------------------------------------------------------------------------- #
# Utility: training loop
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def train_autoencoder(
    model: QuanvolutionAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Training loop for the hybrid quantum‑classical autoencoder.

    Parameters
    ----------
    model : QuanvolutionAutoencoder
        The model to train.
    data : torch.Tensor
        Dataset of shape (N, 1, 28, 28).
    epochs : int, optional
        Number of epochs, by default 100.
    batch_size : int, optional
        Batch size, by default 64.
    lr : float, optional
        Learning rate, by default 1e-3.
    weight_decay : float, optional
        L2 regularization, by default 0.0.
    device : torch.device | None, optional
        Training device; defaults to CUDA if available.

    Returns
    -------
    list[float]
        List of epoch‑level reconstruction losses.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["QuanvolutionAutoencoder", "train_autoencoder"]
