"""EstimatorQNNGen025: classical neural network that outputs quantum parameters.

The network mirrors the simple feed‑forward design of the original
EstimatorQNN but returns two scalars that can be used directly as
parameters for the quantum circuit.  It also implements a batched
evaluation routine with optional Gaussian shot noise, inspired by
FastEstimator.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable, Sequence, List, Callable, Tuple, Any

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class EstimatorQNNGen025(nn.Module):
    """
    Classical neural network that outputs the two parameters required
    by the quantum EstimatorQNNGen025.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 2),  # outputs [input_param, weight_param]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    def to_params(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a batch of inputs into the two quantum parameters.
        """
        out = self.forward(inputs)
        return out[:, 0], out[:, 1]

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of scalar observables on the outputs of the network.
        Supports optional Gaussian shot noise to mimic quantum measurement
        statistics.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["EstimatorQNNGen025"]
