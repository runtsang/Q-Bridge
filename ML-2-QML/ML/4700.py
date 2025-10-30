from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable, List, Callable

# Seed modules
from FCL import FCL
from Conv import Conv

class HybridLayer(nn.Module):
    """
    Hybrid classical layer combining a convolutional filter, a fully connected
    transformation, and optional noisy evaluation using FastEstimator.
    Designed as a drop‑in replacement for the original FCL module while
    exposing the Conv and FastEstimator capabilities from the reference pairs.
    """

    def __init__(self, kernel_size: int = 2, n_features: int = 1, shots: int | None = None, seed: int | None = None):
        super().__init__()
        self.conv = Conv()
        self.fcl = FCL()
        self.kernel_size = kernel_size
        self.n_features = n_features
        self.shots = shots
        self.seed = seed

    def run(self, data: np.ndarray, thetas: Sequence[float]) -> np.ndarray:
        """
        Forward pass through the hybrid layer.

        Parameters
        ----------
        data : np.ndarray
            Input data for the convolutional filter.
        thetas : Sequence[float]
            Parameters for the fully connected layer.

        Returns
        -------
        np.ndarray
            Scalar expectation value mimicking the original FCL implementation.
        """
        conv_out = self.conv.run(data)
        fcl_out = self.fcl.run(thetas)
        return conv_out + fcl_out

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate the layer for multiple parameter sets without shot noise.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Observables to apply to the model output.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter sets for the fully connected layer.

        Returns
        -------
        List[List[float]]
            List of observable values for each parameter set.
        """
        results: List[List[float]] = []
        for params in parameter_sets:
            out = self.fcl.run(params)[0]
            out_tensor = torch.tensor(out, dtype=torch.float32)
            row: List[float] = []
            for obs in observables:
                val = obs(out_tensor)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate with Gaussian shot noise.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Observables to apply to the model output.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter sets for the fully connected layer.
        shots : int
            Number of shots for noisy estimation.
        seed : int, optional
            Random seed for shot noise.

        Returns
        -------
        List[List[float]]
            Noisy observable values for each parameter set.
        """
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        for params in parameter_sets:
            out = self.fcl.run(params)[0]
            out_tensor = torch.tensor(out, dtype=torch.float32)
            row: List[float] = []
            for obs in observables:
                val = obs(out_tensor)
                if isinstance(val, torch.Tensor):
                    mean = float(val.mean().cpu())
                else:
                    mean = float(val)
                noisy = rng.normal(mean, max(1e-6, 1 / shots))
                row.append(noisy)
            results.append(row)
        return results

__all__ = ["HybridLayer"]
