"""Unified FastEstimator combining classical, quantum kernels and sampler support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Any, Optional

import numpy as np
import torch
from torch import nn

# ----- Utility ----------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# ----- Classical / Kernel ----------
class KernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz used for classical kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets of vectors using RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----- Sampler QNN ----------
class SamplerQNN(nn.Module):
    """Simple feed‑forward network that outputs a probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.nn.functional.softmax(self.net(inputs), dim=-1)

def SamplerQNN_factory() -> nn.Module:
    """Return a fresh instance of the sampler network."""
    return SamplerQNN()

# ----- Unified Estimator ----------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastGenEstimator:
    """
    Unified estimator that can work with a classical neural net, an RBF kernel,
    or a sampler network.  For quantum kernels the quantum implementation is
    provided in the QML module.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        # Detect mode
        if isinstance(model, Kernel):
            self._mode = "kernel"
        elif isinstance(model, SamplerQNN):
            self._mode = "sampler"
        else:
            self._mode = "nn"

    # --- Core evaluation ------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        For *nn*: compute observables on model outputs.
        For *kernel*: compute Gram matrix, ignoring ``observables``.
        For *sampler*: return the probability distribution for each param set.
        """
        results: List[List[float]] = []
        if self._mode == "kernel":
            # Treat each parameter set as a data point
            X = [torch.tensor(p, dtype=torch.float32) for p in parameter_sets]
            Y = X  # symmetric Gram matrix
            gram = kernel_matrix(X, Y, gamma=getattr(self.model.ansatz, "gamma", 1.0))
            return gram.tolist()

        if self._mode == "sampler":
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inp = _ensure_batch(params)
                    probs = self.model(inp).squeeze().cpu().numpy().tolist()
                    results.append([float(p) for p in probs])
            return results

        # --- Neural net mode ----------------------------------------------------
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        # Add shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    # --- Convenience ----------------------------------------
    def kernel_matrix(
        self,
        data_a: Sequence[Sequence[float]],
        data_b: Sequence[Sequence[float]],
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Direct access to kernel matrix calculation for kernel mode."""
        if self._mode!= "kernel":
            raise RuntimeError("kernel_matrix is only available for Kernel models")
        X = [torch.tensor(x, dtype=torch.float32) for x in data_a]
        Y = [torch.tensor(y, dtype=torch.float32) for y in data_b]
        return kernel_matrix(X, Y, gamma)

__all__ = [
    "FastGenEstimator",
    "Kernel",
    "KernalAnsatz",
    "kernel_matrix",
    "SamplerQNN",
    "SamplerQNN_factory",
]
