import torch
import numpy as np
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastHybridEstimator:
    """
    Hybrid estimator that evaluates a PyTorch model on batched parameters
    and optionally injects Gaussian shot noise.  It also exposes a
    ``predict`` helper that returns a NumPy array for downstream use.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [lambda out: out.mean(dim=-1)])
        results: List[List[float]] = []
        for params in parameter_sets:
            batch = _ensure_batch(params)
            outputs = self._forward(batch)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> np.ndarray:
        """Convenience wrapper that returns a NumPy array of predictions."""
        return np.array(self.evaluate(
            observables=[lambda out: out.mean(dim=-1)],
            parameter_sets=parameter_sets,
        ), dtype=np.float32)
