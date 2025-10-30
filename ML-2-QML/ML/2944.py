"""Hybrid estimator for classical neural networks with optional auto‑encoding and Gaussian shot noise.

The class extends the original FastBaseEstimator by adding an optional
auto‑encoder preprocessing stage and a flexible observable interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D float32 tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class AutoencoderNet(nn.Module):
    """Simple fully‑connected auto‑encoder used for optional preprocessing."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
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
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class HybridEstimator:
    """Evaluate a PyTorch model (optionally preceded by an auto‑encoder) for
    batches of inputs and observables, with optional Gaussian shot noise.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        autoencoder: Optional[AutoencoderNet] = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.autoencoder = autoencoder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        if self.autoencoder is not None:
            self.autoencoder.to(self.device)

    def _prepare(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pre‑process inputs with the auto‑encoder if supplied."""
        if self.autoencoder is None:
            return inputs
        return self.autoencoder.encode(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables:
            Callables that map a model output tensor to a scalar.
            If empty, the mean of the output across the last dimension is returned.
        parameter_sets:
            Iterable of parameter vectors that will be fed to the model.
        shots, seed:
            If shots is not None, Gaussian noise with std = 1/sqrt(shots) is added
            to each deterministic output.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        if self.autoencoder is not None:
            self.autoencoder.eval()

        rng = np.random.default_rng(seed)

        for params in parameter_sets:
            batch = _ensure_batch(params).to(self.device)
            batch = self._prepare(batch)

            with torch.no_grad():
                output = self.model(batch)

            row: List[float] = []
            for obs in observables:
                val = obs(output)
                scalar = (
                    float(val.mean().cpu())
                    if isinstance(val, torch.Tensor)
                    else float(val)
                )
                row.append(scalar)

            if shots is not None:
                noise = rng.normal(
                    0.0, max(1e-6, 1 / shots), size=len(row)
                )
                row = (np.array(row) + noise).tolist()

            results.append(row)

        return results


__all__ = ["HybridEstimator", "AutoencoderNet"]
