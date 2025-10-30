"""Hybrid estimator integrating classical neural networks, RBF kernels, and convolution filters."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Iterable as IterableType

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a classical model, optionally add shot noise, compute RBF kernels, and apply conv filters."""

    def __init__(self, model: nn.Module | None = None) -> None:
        self.model = model

    def evaluate(
        self,
        observables: IterableType[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for each parameter set and apply the observables.

        Parameters
        ----------
        observables: Iterable[Callable[[Tensor], Tensor | float]]
            Functions that map the network output to a scalar.
        parameter_sets: Sequence[Sequence[float]]
            List of parameter vectors for the model.
        shots: int, optional
            If supplied, Gaussian shot noise with variance 1/shots is added.
        seed: int, optional
            Random seed for reproducibility of shot noise.

        Returns
        -------
        List[List[float]]
            Rows correspond to each parameter set, columns to each observable.
        """
        if self.model is None:
            raise ValueError("HybridEstimator requires a model.")
        if parameter_sets is None:
            raise ValueError("parameter_sets must be supplied.")
        observables = list(observables or [lambda out: out.mean(dim=-1)])
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Classification helper (matches the ML build_classifier_circuit)
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """Construct a feed‑forward classifier and return metadata like the quantum API."""
        layers: list[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: list[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    # ------------------------------------------------------------------
    # RBF kernel utilities
    # ------------------------------------------------------------------
    @staticmethod
    def rbf_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        """Compute the Gram matrix using an RBF kernel."""
        a = torch.stack([x.reshape(1, -1) for x in a])
        b = torch.stack([y.reshape(1, -1) for y in b])
        diff = a - b.transpose(0, 1)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        k = torch.exp(-gamma * dist_sq)
        return k.numpy()

    # ------------------------------------------------------------------
    # Convolution filter
    # ------------------------------------------------------------------
    @staticmethod
    def conv_filter(kernel_size: int = 2, threshold: float = 0.0) -> Callable[[torch.Tensor], float]:
        """Return a callable that applies a 2‑D convolution and sigmoid threshold."""
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(data: torch.Tensor) -> float:
            if data.ndim == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            logits = conv(data)
            activations = torch.sigmoid(logits - threshold)
            return activations.mean().item()

        return run


__all__ = ["HybridEstimator"]
