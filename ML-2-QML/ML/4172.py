"""Combined classical estimator with convolutional filter and kernel support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

# ----------------------------------------------------------------------
# Convolutional filter (drop‑in for quanvolution)
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """2‑D convolutional filter with sigmoid activation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Apply the filter to a 2‑D array and return the mean activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# ----------------------------------------------------------------------
# RBF kernel ansatz
# ----------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Radial‑basis‑function kernel expressed as a torch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that prepares inputs for the kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Combined estimator
# ----------------------------------------------------------------------
ScalarObservable = Callable[[Union[torch.Tensor, np.ndarray]], torch.Tensor | float | np.ndarray]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimatorGen152:
    """Unified estimator that can evaluate either a torch model, a convolutional filter, or a kernel ansatz.  Supports optional Gaussian shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute scalar outputs for each parameter set and observable."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Dispatch based on model type
                if hasattr(self.model, "run"):
                    # Convolutional filter expects raw data
                    outputs = self.model.run(np.asarray(params))
                else:
                    inputs = _ensure_batch(params)
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

        # Add shot‑noise if requested
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["ConvFilter", "KernalAnsatz", "Kernel", "kernel_matrix", "FastBaseEstimatorGen152"]
