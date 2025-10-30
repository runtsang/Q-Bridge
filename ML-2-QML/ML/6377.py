import torch
import torch.nn as nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters to a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridBaseEstimator:
    """Fast classical estimator for a `torch.nn.Module`.

    Evaluates a model on batches of input parameters and applies one or
    more scalar observables.  Optionally adds Gaussian shot noise to
    mimic a noisy measurement.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables)
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]

        raw: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                raw.append(row)

        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridBaseEstimator"]
