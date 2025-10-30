"""Hybrid quantum kernel module with classical backends.

The module defines :class:`HybridQuantumKernel` which exposes a unified interface
for computing classical RBF kernels, evaluating fast estimators, sampling from
parameterized networks, and running a fully‑connected quantum‑style layer.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Callable

import numpy as np
import torch
from torch import nn

# ---------- Classical RBF kernel ----------
class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    k = RBFKernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# ---------- Fast estimators ----------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
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
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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

# ---------- Sampler network ----------
def SamplerQNN() -> nn.Module:
    """Simple feed‑forward network that produces a probability distribution."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

# ---------- Fully‑connected layer ----------
def FCL() -> nn.Module:
    """Classical stand‑in for a quantum fully‑connected layer."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()

# ---------- Hybrid kernel class ----------
class HybridQuantumKernel:
    """
    Unified interface for classical and quantum kernel operations.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width (used only for the classical kernel).
    n_wires : int, optional
        Number of qubits for the quantum kernel (unused in the classical path).
    use_sampler : bool, optional
        Whether to expose a sampler network.
    use_fcl : bool, optional
        Whether to expose a fully‑connected layer.
    """
    def __init__(self, gamma: float = 1.0, n_wires: int = 4,
                 use_sampler: bool = False, use_fcl: bool = False) -> None:
        self.gamma = gamma
        self.rbf = RBFKernel(gamma)
        self.use_sampler = use_sampler
        self.use_fcl = use_fcl
        if use_sampler:
            self.sampler = SamplerQNN()
        if use_fcl:
            self.fcl = FCL()

    # Classical kernel
    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rbf(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, self.gamma)

    # Fast estimator wrapper
    def evaluate(
        self,
        model: nn.Module,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        estimator = FastEstimator(model) if shots is not None else FastBaseEstimator(model)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

    # Sampler
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled for this kernel.")
        return self.sampler(inputs)

    # Fully‑connected layer
    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        if not self.use_fcl:
            raise RuntimeError("Fully‑connected layer not enabled.")
        return self.fcl.run(thetas)

__all__ = [
    "HybridQuantumKernel",
    "RBFKernel",
    "FastBaseEstimator",
    "FastEstimator",
    "SamplerQNN",
    "FCL",
    "kernel_matrix",
]
