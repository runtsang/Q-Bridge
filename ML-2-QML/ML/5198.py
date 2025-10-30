"""Hybrid estimator supporting both PyTorch neural networks and quantum circuits.

The class combines the lightweight batch evaluation of FastBaseEstimator with
shot‑noise simulation from FastEstimator.  It accepts any callable as an
observable, allowing users to plug in convolution filters or self‑attention
transformations.

Example usage:

    from FastBaseEstimator__gen078 import FastHybridEstimator

    model = nn.Linear(10, 1)
    estimator = FastHybridEstimator(model, shots=1000, seed=42)

    observables = [lambda out: out.mean(dim=-1),
                   Conv().run,
                   SelfAttention().run]
    predictions = estimator.evaluate(observables, parameter_sets=[[0.1]*10])

"""

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters into a batched tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastHybridEstimator:
    """Evaluate a PyTorch model for a sequence of parameter sets.

    Parameters
    ----------
    model
        A ``torch.nn.Module`` that maps a batch of parameter vectors to
        output tensors.
    shots
        If provided, Gaussian shot noise with variance 1/shots is added to
        each observable value.
    seed
        Random seed for reproducible noise.
    """

    def __init__(self, model: nn.Module, *, shots: int | None = None, seed: int | None = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Each row corresponds to one parameter set and each column to one
        observable.  Observables are callables that take the model output and
        return a tensor or scalar.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
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

        if self.shots is None:
            return results

        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastHybridEstimator"]
