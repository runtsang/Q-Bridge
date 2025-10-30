from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Hybrid estimator that evaluates a PyTorch model for many parameter sets and observables.

    The estimator supports deterministic evaluation, optional shot‑like Gaussian noise and
    analytic gradients via PyTorch autograd.  It keeps the lightweight API of the original
    FastBaseEstimator but adds useful quantum‑inspired features that can be useful
    for hybrid classical‑quantum workflows.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of callables that map a model output tensor to a scalar.
        parameter_sets
            Sequence of parameter vectors, each a sequence of floats.
        shots
            Optional number of “shots” to add Gaussian noise to the output.
        seed
            Random seed for reproducibility of the noise.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
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
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[torch.Tensor]]]:
        """
        Compute analytic gradients of each observable w.r.t. all trainable parameters
        for each parameter set.

        Returns
        -------
        List[List[List[torch.Tensor]]]
            For each parameter set and each observable a list of gradients (one per
            model parameter).  Gradients are detached tensors.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads_per_set: List[List[List[torch.Tensor]]] = []

        for params in parameter_sets:
            param_tensor = _ensure_batch(params).requires_grad_(True)
            outputs = self.model(param_tensor)
            grads_for_set: List[List[torch.Tensor]] = []

            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    v = value
                else:
                    v = torch.tensor(value, dtype=outputs.dtype, device=outputs.device)
                grads = torch.autograd.grad(
                    v, self.model.parameters(), retain_graph=True, allow_unused=True
                )
                grads_for_set.append(
                    [g.detach() if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, self.model.parameters())]
                )
            grads_per_set.append(grads_for_set)

        return grads_per_set


__all__ = ["FastHybridEstimator"]
