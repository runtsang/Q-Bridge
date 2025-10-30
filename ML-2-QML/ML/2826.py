import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of numerical values into a 1‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter that emulates a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, 1, H, W) or (1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class FastBaseEstimator:
    """Evaluate a torch model for batches of inputs and scalar observables."""
    def __init__(self, model: nn.Module):
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
                # Convert each flat patch to a tensor of shape
                # (batch, 1, kernel, kernel)
                inputs = _ensure_batch(params)
                inputs = inputs.reshape(-1, 1, self.model.kernel_size, self.model.kernel_size)
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
    """Adds Gaussian shot noise to the deterministic estimator."""
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

class ConvEstimator:
    """
    Unified estimator that can run a classical convolutional filter
    or a quantum quanvolution circuit.  The class exposes a single
    ``evaluate`` method that accepts a list of input parameter sets
    (each a flattened 2‑D patch) and an optional list of scalar
    observables.  When ``shots`` is provided the estimator injects
    shot noise to mimic a real quantum backend.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.filter = ConvFilter(kernel_size, threshold)
        self.estimator = FastEstimator(self.filter) if shots is not None else FastBaseEstimator(self.filter)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable] | None = None,
    ) -> List[List[float]]:
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        if isinstance(self.estimator, FastEstimator):
            return self.estimator.evaluate(observables, parameter_sets, shots=self.shots, seed=self.seed)
        else:
            return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["ConvEstimator"]
