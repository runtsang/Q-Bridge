import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate neural networks on batched inputs with arbitrary observables.

    The estimator is device‑agnostic and supports optional batching for large
    parameter sets. Observables are callables that operate on the model
    output and return a scalar or a Tensor.  The class also exposes a
    convenience method for computing analytical gradients via
    ``torch.autograd``.
    """

    def __init__(self, model: nn.Module, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Parameters
        ----------
        observables
            Iterable of callables that take the model output and return a
            scalar or Tensor.  If empty, the default is the mean of the
            output.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
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
        return results

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """Return analytical gradients of each observable w.r.t. inputs.

        The gradients are returned as a list of lists of tensors with shape
        matching the input parameters.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []
        for params in parameter_sets:
            batch = _ensure_batch(params).requires_grad_(True)
            outputs = self._forward(batch)
            row: List[torch.Tensor] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                val.backward(retain_graph=True)
                row.append(batch.grad.clone())
                batch.grad.zero_()
            grads.append(row)
        return grads

class FastEstimator(FastBaseEstimator):
    """Same as :class:`FastBaseEstimator` but adds optional Gaussian shot noise.

    The ``shots`` argument controls the variance ``σ²=1/shots`` of the noise.
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
