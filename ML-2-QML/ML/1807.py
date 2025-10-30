import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Callable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """
    Evaluate a PyTorch model for multiple parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, optional
        Computation device; defaults to CPU.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.model(params.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return scalar results for each observable and parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """
    Adds Gaussian shot noise to deterministic evaluations.

    Parameters
    ----------
    shots : int | None, optional
        Number of shots; if None, no noise is added.
    seed : int | None, optional
        RNG seed for reproducibility.
    """

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

__all__ = ["FastBaseEstimator", "FastEstimator"]
