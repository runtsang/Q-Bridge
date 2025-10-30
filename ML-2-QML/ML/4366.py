"""Hybrid classical autoencoder with QCNN-inspired convolution and graph-based latent adjacency.

This module combines the convolutional stack from the original QCNN model, the
autoencoder architecture from the Autoencoder seed, and the graph adjacency
construction from GraphQNN.  It also exposes a lightweight estimator that
mirrors the FastBaseEstimator for fast batch evaluation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, List, Sequence, Tuple, Callable

# ---------------------------------------------------------------------------

class QCNNHybridAutoencoder(nn.Module):
    """
    A hybrid network that first applies a QCNN-like convolutional stack,
    then encodes the result into a latent space, decodes it back, and
    finally computes a graph adjacency on the latent vectors.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Feature map and QCNN layers
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Autoencoder part
        encoder_layers = []
        in_dim = 4
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, 4))
        self.decoder = nn.Sequential(*decoder_layers)

        # Final classification head (optional)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # QCNN stack
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Autoencoder
        latent = self.encode(x)
        reconstructed = self.decode(latent)

        # Classification
        out = torch.sigmoid(self.head(reconstructed))
        return out

    # -----------------------------------------------------------------------
    # Autoencoder helpers
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # -----------------------------------------------------------------------
    # Graph adjacency on latent space
    def latent_fidelity(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        """Cosine similarity as a proxy for fidelity."""
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-12)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True) + 1e-12)
        return float((z1_norm * z2_norm).sum(dim=-1).mean().item())

    def fidelity_adjacency(
        self,
        latent_vectors: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Return an adjacency matrix (torch.Tensor) where entries are 1.0,
        secondary_weight, or 0.0 based on pairwise fidelities.
        """
        n = len(latent_vectors)
        adj = torch.zeros((n, n), dtype=torch.float32, device=latent_vectors[0].device)
        for i in range(n):
            for j in range(i + 1, n):
                fid = self.latent_fidelity(latent_vectors[i], latent_vectors[j])
                if fid >= threshold:
                    adj[i, j] = adj[j, i] = 1.0
                elif secondary is not None and fid >= secondary:
                    adj[i, j] = adj[j, i] = secondary_weight
        return adj

# ---------------------------------------------------------------------------

def QCNNHybridAutoencoderFactory(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> QCNNHybridAutoencoder:
    """Return a configured hybrid autoencoder."""
    return QCNNHybridAutoencoder(
        input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout
    )

# ---------------------------------------------------------------------------

def train_autoencoder(
    model: QCNNHybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
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

# ---------------------------------------------------------------------------

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# ---------------------------------------------------------------------------

# Lightweight estimator (FastBaseEstimator style)
class FastHybridEstimator:
    """
    Evaluates a QCNNHybridAutoencoder on batches of inputs with optional shot noise.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        inputs: Sequence[torch.Tensor],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        self.model.eval()
        device = next(self.model.parameters()).device
        batch = torch.stack([x.to(device) for x in inputs])
        with torch.no_grad():
            outputs = self.model(batch)
            results = outputs.cpu().tolist()
        if shots is None:
            return results
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        noisy = []
        for row in results:
            noisy_row = [float(rng.normal_(torch.tensor(val), 1 / shots).item()) for val in row]
            noisy.append(noisy_row)
        return noisy

# ---------------------------------------------------------------------------

__all__ = [
    "QCNNHybridAutoencoder",
    "QCNNHybridAutoencoderFactory",
    "train_autoencoder",
    "FastHybridEstimator",
]
