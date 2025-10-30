"""Combined classical estimator integrating regression, sampling, and kernel methods."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Iterable, List, Sequence, Union

# ------------------------------------------------------------------
# Utility estimators
# ------------------------------------------------------------------
class FastEstimator:
    """Deterministic estimator with optional shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Union[torch.Tensor, float]]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
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
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

# ------------------------------------------------------------------
# Classical sub‑modules
# ------------------------------------------------------------------
class SamplerModule(nn.Module):
    """Softmax sampler network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class KernalAnsatz(nn.Module):
    """RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper for the RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Unified estimator
# ------------------------------------------------------------------
class SharedEstimator(nn.Module):
    """Unified classical estimator combining regression, sampling and kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        # Regression backbone
        self.regressor = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Sampler head
        self.sampler = SamplerModule()
        # Kernel module
        self.kernel = Kernel(gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(inputs)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return regression predictions."""
        return self.forward(inputs)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over two classes."""
        return self.sampler(inputs)

    def kernel_between(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the embedded RBF kernel."""
        return kernel_matrix(a, b, gamma=self.kernel.gamma)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Union[torch.Tensor, float]]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["SharedEstimator", "kernel_matrix"]
