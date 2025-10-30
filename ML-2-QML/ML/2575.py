"""Hybrid classical kernel evaluator combining RBF and neural‑network features with efficient batch evaluation."""

import numpy as np
import torch
from torch import nn
from typing import Sequence, Callable, List, Iterable

# Classical RBF kernel components
class KernalAnsatz(nn.Module):
    """RBF kernel ansatz with a fixed gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps KernalAnsatz for pairwise kernel evaluation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix_classical(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Fast estimator utilities
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Batch evaluator for a torch model with optional observables."""
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
    """Adds Gaussian shot noise to deterministic estimates."""
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

# Hybrid kernel evaluator
class HybridKernelEvaluator:
    """Combines classical RBF and a user‑supplied torch model kernel into a weighted kernel."""
    def __init__(self, classical_gamma: float = 1.0, quantum_weight: float = 0.5) -> None:
        self.classical_gamma = classical_gamma
        self.quantum_weight = quantum_weight
        self.classical_kernel = Kernel(classical_gamma)

    def classical_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix_classical(a, b, self.classical_gamma)

    def quantum_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], model: nn.Module) -> np.ndarray:
        """Assumes model outputs a scalar kernel value for each pair."""
        estimator = FastEstimator(model)
        # Flatten pairs
        pairs = [(x, y) for x in a for y in b]
        observables = [lambda out: out]  # model outputs scalar kernel
        params = [torch.cat([x, y]).tolist() for x, y in pairs]
        results = estimator.evaluate(observables, params)
        mat = np.array([row[0] for row in results]).reshape(len(a), len(b))
        return mat

    def hybrid_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], model: nn.Module) -> np.ndarray:
        Kc = self.classical_kernel_matrix(a, b)
        Kq = self.quantum_kernel_matrix(a, b, model)
        return self.quantum_weight * Kq + (1 - self.quantum_weight) * Kc

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix_classical",
    "FastBaseEstimator",
    "FastEstimator",
    "HybridKernelEvaluator",
]
