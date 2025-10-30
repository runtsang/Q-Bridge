"""Hybrid classical autoencoder with optional quantum-inspired components and fast evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------
# Fast estimator utilities (adapted from the FastBaseEstimator pair)
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# ----------------------------------------------------------------------
# Quanvolution-inspired filter
# ----------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution followed by flattening, mimicking a quantum kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


# ----------------------------------------------------------------------
# Hybrid autoencoder architecture
# ----------------------------------------------------------------------
@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_shape: Tuple[int, int, int]  # (channels, height, width)
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quanvolution: bool = True


class HybridAutoencoder(nn.Module):
    """Convolutional encoder (optionally quanvolutional) + fully‑connected decoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        if config.use_quanvolution:
            self.encoder = nn.Sequential(
                QuanvolutionFilter(*config.input_shape),
                nn.Linear(
                    np.prod(config.input_shape) // (config.input_shape[1] // 2) // (config.input_shape[2] // 2),
                    config.hidden_dims[0],
                ),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(config.input_shape), config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )

        # Latent projection
        self.latent = nn.Linear(config.hidden_dims[0], config.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], np.prod(config.input_shape)),
            nn.Unflatten(1, config.input_shape),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.latent(self.encoder(inputs))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


# ----------------------------------------------------------------------
# Synthetic dataset (from the QuantumRegression pair)
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class SuperpositionDataset(Dataset):
    """Dataset that returns raw feature vectors for auto‑encoding."""
    def __init__(self, samples: int, num_features: int) -> None:
        self.features, _ = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(self.features[index], dtype=torch.float32)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(data_loader.dataset)
        history.append(epoch_loss)
    return history


# ----------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------
def evaluate_autoencoder(
    model: HybridAutoencoder,
    inputs: torch.Tensor,
    *,
    observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
    noise_shots: int | None = None,
    seed: int | None = None,
) -> List[List[float]]:
    estimator_cls = FastEstimator if noise_shots is not None else FastBaseEstimator
    estimator = estimator_cls(model)
    observable_list = list(observables) if observables else [lambda outputs: outputs.mean(dim=-1)]
    parameter_sets = [inputs.squeeze().tolist()]
    return estimator.evaluate(observable_list, parameter_sets, shots=noise_shots, seed=seed) if noise_shots else estimator.evaluate(observable_list, parameter_sets)


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
    "evaluate_autoencoder",
    "generate_superposition_data",
    "SuperpositionDataset",
]
