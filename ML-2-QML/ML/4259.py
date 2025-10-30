"""Hybrid classical estimator that blends autoencoding and quantum‑inspired layers.

The architecture mirrors the classical AutoencoderNet encoder/decoder
while inserting a linear “quantum” block that emulates the behaviour
of a variational circuit.  The dataset is generated from a
superposition distribution used in the quantum reference, enabling
direct comparison between classical and quantum training pipelines."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data: x in [-1,1]^d, y = sin(sum(x)) + 0.1*cos(2*sum(x))."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class SuperpositionDataset(Dataset):
    """Dataset exposing feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridEstimatorQNN(nn.Module):
    """Classical network with an autoencoder skeleton and a quantum‑inspired linear block."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, int] = (64, 32),
    ) -> None:
        super().__init__()
        # Encoder (mirrors AutoencoderNet encoder)
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum‑inspired linear block (three linear layers with ReLU)
        self.quantum_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Decoder (mirrors AutoencoderNet decoder)
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encoder(x)
        qz = self.quantum_layer(z)
        out = self.decoder(qz)
        return out.squeeze(-1)


def train_hybrid(
    model: HybridEstimatorQNN,
    dataset: SuperpositionDataset,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid model on the superposition dataset.

    Returns a list of training losses per epoch."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(states)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridEstimatorQNN",
    "SuperpositionDataset",
    "generate_superposition_data",
    "train_hybrid",
]
