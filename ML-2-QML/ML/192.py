import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[Sequence[float]]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimatorGen212:
    """Vectorised deterministic estimator for PyTorch models."""
    def __init__(self, model: nn.Module, *, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
    ) -> List[List[float]]:
        """Compute expectation values for all parameter sets in batches."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = [[] for _ in parameter_sets]
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(parameter_sets), batch_size):
                batch = parameter_sets[i : i + batch_size]
                inputs = _ensure_batch(batch).to(self.device)
                outputs = self.model(inputs)
                for obs in observables:
                    vals = obs(outputs)
                    if isinstance(vals, torch.Tensor):
                        vals = vals.cpu().tolist()
                    else:
                        vals = [float(vals)] * len(batch)
                    for j, val in enumerate(vals):
                        results[i + j].append(val)
        return results

class FastEstimatorGen212(FastBaseEstimatorGen212):
    """Adds Gaussian shot‑noise to the deterministic predictions."""
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

__all__ = ["FastBaseEstimatorGen212", "FastEstimatorGen212"]
