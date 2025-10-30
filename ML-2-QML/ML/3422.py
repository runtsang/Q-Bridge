"""Hybrid classical autoencoder with optional quantum latent representation.

The module defines a lightweight MLP autoencoder, training utilities that can
integrate a quantum encoder, and a simple noise injection helper inspired
by the FastEstimator/ FastBaseEstimator patterns from the reference seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# Utility helpers (FastEstimator / FastBaseEstimator inspired)
# ---------------------------------------------------------------------


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D float32 tensor suitable for batch processing."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _add_gaussian_noise(values: Iterable[float], shots: int | None = None, seed: int | None = None) -> List[float]:
    """Inject Gaussian shot noise into a list of scalar values."""
    if shots is None:
        return list(values)
    rng = np.random.default_rng(seed)
    noisy = [float(rng.normal(v, max(1e-6, 1 / shots))) for v in values]
    return noisy


# ---------------------------------------------------------------------
# Autoencoder definition
# ---------------------------------------------------------------------


@dataclass
class AutoencoderConfig:
    """Configuration for the classical autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderHybridNet(nn.Module):
    """A lightweight MLP autoencoder with optional quantum latent input."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """End‑to‑end reconstruction."""
        return self.decode(self.encode(inputs))


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------


def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    quantum_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    noise_shots: int | None = None,
    noise_seed: int | None = None,
) -> List[float]:
    """Train the hybrid autoencoder.

    Parameters
    ----------
    model : AutoencoderHybridNet
        The neural network to train.
    data : torch.Tensor
        Training data (shape: [N, input_dim]).
    quantum_encoder : Callable[[torch.Tensor], torch.Tensor] | None
        Optional callable that maps a batch of inputs to a latent tensor
        using a quantum encoder. If ``None`` the classical encoder is used.
    noise_shots : int | None
        If provided, Gaussian shot noise is applied to the latent vector
        before decoding, mimicking measurement statistics.
    noise_seed : int | None
        Random seed for the Gaussian noise generator.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_ensure_batch(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            if quantum_encoder is not None:
                # Quantum encoder may return a CPU tensor; move to device
                latent = quantum_encoder(batch)
                latent = latent.to(device)
                if noise_shots is not None:
                    # Apply shot noise to each latent dimension
                    noise = torch.tensor(
                        _add_gaussian_noise(
                            latent.cpu().numpy().flatten(),
                            shots=noise_shots,
                            seed=noise_seed,
                        ),
                        dtype=torch.float32,
                    ).reshape(latent.shape).to(device)
                    latent = noise
            else:
                latent = model.encode(batch)

            reconstruction = model.decode(latent)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


# ---------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------


def evaluate_hybrid(
    model: nn.Module,
    inputs: torch.Tensor,
    observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
    *,
    shots: int | None = None,
    seed: int | None = None,
) -> List[List[float]]:
    """Compute observable values for a batch of inputs.

    Parameters
    ----------
    model : nn.Module
        The (classical) model to evaluate. It must expose a ``forward`` method.
    inputs : torch.Tensor
        Batch of inputs (shape: [N, input_dim]).
    observables : Iterable[Callable[[torch.Tensor], torch.Tensor]]
        List of callables that map the model output to a scalar.
    shots : int | None
        If provided, Gaussian shot noise is applied to each observable
        value. This mimics the FastEstimator pattern for quantum measurements.
    seed : int | None
        Random seed for the noise generator.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    results: List[List[float]] = []
    for row in outputs:
        row_values = [float(obs(row)) for obs in observables]
        row_values = _add_gaussian_noise(row_values, shots=shots, seed=seed)
        results.append(row_values)
    return results


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "AutoencoderConfig",
    "AutoencoderHybridNet",
    "train_autoencoder_hybrid",
    "evaluate_hybrid",
    "_add_gaussian_noise",
]
