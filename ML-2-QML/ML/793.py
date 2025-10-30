"""Advanced hybrid estimator with Bayesian regularisation and adaptive sampling.

This module builds on the original lightweight estimator by adding a neural network backbone
for feature extraction, a Bayesian linear layer for output prediction, and an optional
adaptive sampling routine that uses the prior posterior to decide which parameter
set to evaluate next.  The design is deliberately kept modular so that the user can
swap out the backbone or the quantum module without changing the API.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
# Core estimator class
# --------------------------------------------------------------------------- #
class AdvancedHybridEstimator:
    """Evaluate neural networks for batches of inputs and observables,
    optionally adding Gaussian shot noise and performing adaptive sampling.

    Parameters
    ----------
    backbone : nn.Module
        Feature extractor that maps raw input parameters to a latent space.
    bayes_layer : nn.Module
        Bayesian linear layer that produces a predictive distribution
        given the latent representation.  The module must expose a
        ``forward`` method returning a tuple ``(mean, logvar)``.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise applied to the
        observable outputs.  ``0`` disables noise.
    device : str, optional
        Target device for the PyTorch tensors.
    """

    def __init__(
        self,
        backbone: nn.Module,
        bayes_layer: nn.Module,
        noise_std: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.backbone = backbone.to(device)
        self.bayes_layer = bayes_layer.to(device)
        self.noise_std = noise_std
        self.device = device
        self.backbone.eval()
        self.bayes_layer.eval()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute observable expectations for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the output of the Bayesian layer
            (typically a mean tensor) and returns a scalar or a
            tensor that can be reduced to a scalar.
        parameter_sets : sequence of sequences
            List of parameter vectors to evaluate.

        Returns
        -------
        List[List[float]]
            A matrix of shape ``(n_params, n_observables)``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params).to(self.device)
                features = self.backbone(batch)
                mean, logvar = self.bayes_layer(features)
                # Draw a single sample from the predictive distribution
                std = torch.exp(0.5 * logvar)
                sample = mean + std * torch.randn_like(std)
                row: List[float] = []
                for obs in observables:
                    value = obs(sample)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    # Inject Gaussian shot noise if requested
                    if self.noise_std > 0.0:
                        scalar += random.gauss(0.0, self.noise_std)
                    row.append(scalar)
                results.append(row)
        return results

    # --------------------------------------------------------------------- #
    # Adaptive sampling
    # --------------------------------------------------------------------- #
    def select_next(
        self,
        prior_mean: torch.Tensor,
        prior_cov: torch.Tensor,
        candidate_sets: Sequence[Sequence[float]],
        metric: Callable[[torch.Tensor], float] = lambda x: x.mean().item(),
    ) -> Sequence[float]:
        """
        Choose the next parameter set to evaluate by maximizing an
        acquisition metric (default: mean prediction).

        Parameters
        ----------
        prior_mean : torch.Tensor
            Current posterior mean over the latent space.
        prior_cov : torch.Tensor
            Current posterior covariance over the latent space.
        candidate_sets : sequence of sequences
            List of parameter vectors to consider.
        metric : callable, optional
            Function that maps a predictive mean tensor to a scalar
            acquisition value.

        Returns
        -------
        Sequence[float]
            The parameter vector with the highest acquisition value.
        """
        best_val = -math.inf
        best_params: Optional[Sequence[float]] = None

        with torch.no_grad():
            for params in candidate_sets:
                batch = _ensure_batch(params).to(self.device)
                features = self.backbone(batch)
                mean, _ = self.bayes_layer(features)
                score = metric(mean)
                if score > best_val:
                    best_val = score
                    best_params = list(params)

        if best_params is None:
            raise RuntimeError("No candidate parameters were supplied.")
        return best_params

__all__ = ["AdvancedHybridEstimator"]
