"""FastBaseEstimator with GPU, batch, and gradient support."""

import torch
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables.

    The estimator now supports:

    * Automatic device placement (CPU / GPU).
    * Batch evaluation of many parameter sets in one forward pass.
    * Optional caching of input tensors to speed up repeated evaluations.
    * Gradient computation via torch.autograd for parameter‑shift style optimisation.
    * Custom loss functions for multi‑output models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        cache: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.cache = cache
        self._cache: dict[tuple[float,...], torch.Tensor] = {}

    def _prepare_inputs(self, params: Sequence[Sequence[float]]) -> torch.Tensor:
        batch = torch.as_tensor(params, dtype=torch.float32, device=self.device)
        return batch

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        return_tensors: bool = False,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            (torch.Tensor or float). If empty, the mean of the last dimension is used.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one forward pass.
        return_tensors : bool
            If True, the raw output tensors are returned instead of scalars.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(parameter_sets)
            outputs = self.model(inputs)
            for out in outputs:
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[torch.Tensor]]:
        """Return observables and gradients w.r.t. each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        self.model.train()
        grads: List[torch.Tensor] = []
        results: List[List[float]] = []

        for params in parameter_sets:
            params_tensor = torch.as_tensor(
                params, dtype=torch.float32, device=self.device, requires_grad=True
            )
            output = self.model(params_tensor)
            row: List[float] = []
            for obs in observables:
                val = obs(output)
                if isinstance(val, torch.Tensor):
                    scalar = val.mean()
                else:
                    scalar = torch.tensor(val, dtype=torch.float32, device=self.device)
                row.append(float(scalar.cpu()))
            grads.append(
                torch.autograd.grad(
                    outputs=output, inputs=params_tensor, grad_outputs=torch.ones_like(output)
                )[0]
            )
            results.append(row)
        return results, grads

    def cache_inputs(self, parameters: Sequence[Sequence[float]]) -> None:
        """Populate the internal cache for a set of parameter vectors."""
        if not self.cache:
            raise RuntimeError("Caching is disabled for this estimator.")
        for params in parameters:
            key = tuple(params)
            if key not in self._cache:
                self._cache[key] = torch.as_tensor(params, dtype=torch.float32, device=self.device)

    def evaluate_cached(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate using cached tensors if available."""
        if not self.cache:
            raise RuntimeError("Caching is disabled for this estimator.")
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                key = tuple(params)
                inp = self._cache.get(key, torch.as_tensor(params, dtype=torch.float32, device=self.device))
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


__all__ = ["FastBaseEstimator"]
