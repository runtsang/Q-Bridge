"""Combined classical kernel and estimator module."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable, Callable, List

# --------------------------------------------------------------------------- #
# Classical kernel utilities
# --------------------------------------------------------------------------- #
class ClassicalKernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz with vectorized support."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Broadcast to compute pairwise differences
        diff = x[:, None, :] - y[None, :, :]
        sq_norm = (diff * diff).sum(-1)
        return torch.exp(-self.gamma * sq_norm)

class ClassicalKernel(nn.Module):
    """Wrapper that exposes a simple forward interface for two vectors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two batches of vectors."""
    kernel = ClassicalKernel(gamma)
    return kernel(torch.stack(a), torch.stack(b)).cpu().numpy()

# --------------------------------------------------------------------------- #
# Fully connected layer (classical) and estimator
# --------------------------------------------------------------------------- #
class FullyConnectedLayer(nn.Module):
    """Simple linear layer followed by tanh activation."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean().item()

class FastBaseEstimator:
    """Deterministic estimator that accepts a model and a set of observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        if not observables:
            observables = [lambda x: x.mean(dim=-1)]
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
    """Adds Gaussian shot noise to the deterministic estimator."""
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

# --------------------------------------------------------------------------- #
# Combined shared model
# --------------------------------------------------------------------------- #
class SharedKernelModel:
    """Unified interface that supports both classical RBF kernels and quantum‑style kernels.
    It also exposes a fully‑connected layer and an estimator that can add shot noise."""
    def __init__(
        self,
        mode: str = "classical",
        gamma: float = 1.0,
        noise_shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.mode = mode
        self.noise_shots = noise_shots
        self.seed = seed
        if mode == "classical":
            self.kernel = ClassicalKernel(gamma)
            self.fcl = FullyConnectedLayer()
        else:
            raise NotImplementedError("Quantum mode is not available in the classical implementation.")
        self.estimator = FastEstimator(self.fcl)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.kernel.ansatz.gamma)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        return self.estimator.evaluate(
            observables, parameter_sets, shots=self.noise_shots, seed=self.seed
        )

__all__ = ["SharedKernelModel"]
