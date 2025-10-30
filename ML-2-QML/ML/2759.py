import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridEstimator:
    """
    Classical estimator that evaluates a PyTorch model over parameter sets and observables.
    Supports optional Gaussian shot noise and batched evaluation for efficiency.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar or a tensor.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters to feed to the model.
        shots : int, optional
            If provided, adds Gaussian noise with variance 1/shots to each result.
        seed : int, optional
            Random seed for reproducibility of noise.

        Returns
        -------
        List[List[float]]
            Nested list of results: outer list over parameter sets, inner list over observables.
        """
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
                if shots is not None:
                    rng = np.random.default_rng(seed)
                    row = [float(rng.normal(m, max(1e-6, 1 / shots))) for m in row]
                results.append(row)
        return results

__all__ = ["HybridEstimator"]
