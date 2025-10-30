import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable, List, Callable

# --- Fast estimator utilities -------------------------------------------------
class FastBaseEstimator:
    """
    Evaluate a torch.nn.Module for a batch of parameter sets.
    """
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
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """
    Adds Gaussian shot noise to the deterministic estimator.
    """
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

# --- Classical RBF kernel -----------------------------------------------------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    k = Kernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# --- Combined ML class --------------------------------------------------------
class QuantumKernelMethod:
    """
    Hybrid interface that exposes a classical RBF kernel and a FastEstimator
    for deterministic or noisy evaluation of a neural network model.
    """
    def __init__(self, model: nn.Module | None = None, gamma: float = 1.0) -> None:
        self.model = model
        self.gamma = gamma
        self.kernel = Kernel(gamma)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, self.gamma)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if self.model is None:
            raise ValueError("No model provided for evaluation.")
        if shots is None:
            estimator = FastBaseEstimator(self.model)
            return estimator.evaluate(observables, parameter_sets)
        else:
            estimator = FastEstimator(self.model)
            return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "QuantumKernelMethod",
    "FastBaseEstimator",
    "FastEstimator",
    "Kernel",
    "KernalAnsatz",
    "kernel_matrix",
]
