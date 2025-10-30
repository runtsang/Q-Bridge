"""Hybrid estimator that can evaluate classical or quantum models with optional autoencoder preprocessing and Gaussian noise.

The class mirrors the original FastBaseEstimator but extends it with:
* optional autoencoder preprocessing (latent space fed to the model)
* optional shot‑noise simulation
* a static autoencoder training helper
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a float32 tensor with a leading batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------------- #
class HybridFastEstimator:
    """Evaluate a PyTorch model (classical or hybrid) with optional preprocessing and noise.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model that accepts a batch of parameters and returns outputs.
    autoencoder : nn.Module | None, optional
        If provided, its ``encode`` output is fed into ``model``.
    add_noise : bool, default=False
        Whether to add Gaussian shot noise to the results.
    shots : int | None, optional
        Number of shots for noise simulation; ignored if ``add_noise`` is False.
    seed : int | None, optional
        Random seed for reproducibility of the noise generator.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        autoencoder: nn.Module | None = None,
        add_noise: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.autoencoder = autoencoder
        self.add_noise = add_noise
        self.shots = shots
        self.seed = seed

    # --------------------------------------------------------------------- #
    # Pre‑processing
    # --------------------------------------------------------------------- #
    def _preprocess(self, params: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            return params
        with torch.no_grad():
            latent = self.autoencoder.encode(params)
        return latent

    # --------------------------------------------------------------------- #
    # Evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable]
            Functions that map model outputs to scalar values.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[float]]
            Nested list of scalar results.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                processed = self._preprocess(inputs)
                outputs = self.model(processed)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if self.add_noise and self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    # --------------------------------------------------------------------- #
    # Autoencoder helper
    # --------------------------------------------------------------------- #
    @staticmethod
    def train_autoencoder(
        autoencoder: nn.Module,
        data: torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> list[float]:
        """Train an autoencoder and return the loss history."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder.to(device)
        dataset = torch.utils.data.TensorDataset(_ensure_batch(data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

__all__ = ["HybridFastEstimator"]
