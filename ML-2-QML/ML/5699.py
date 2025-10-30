"""Advanced estimator utilities implemented with PyTorch modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class AdvancedBaseEstimator:
    """
    Evaluate neural networks for batches of inputs and observables.

    Features
    --------
    * GPU support via ``device`` argument.
    * Optional automatic differentiation to obtain gradients w.r.t. inputs.
    * Batched evaluation for efficiency.
    * ``add_noise`` method for Gaussian shot noise simulation.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        compute_gradients: bool = False,
        grad_keys: Optional[Sequence[int]] = None,
    ) -> Tuple[List[List[float]], Optional[List[List[torch.Tensor]]]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable accepts the model output and returns a scalar tensor or value.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is a list of floats to feed the model.
        compute_gradients : bool, optional
            If True, compute gradients of each observable w.r.t. the specified inputs.
        grad_keys : sequence of int, optional
            Indices of parameters to differentiate with respect to. If None and
            ``compute_gradients`` is True, gradients w.r.t. all inputs are returned.

        Returns
        -------
        results : list of list of float
            Observable values for each parameter set.
        gradients : list of list of torch.Tensor, optional
            Gradients for each observable w.r.t. the selected inputs.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        gradients: List[List[torch.Tensor]] | None = None

        self.model.eval()
        with torch.set_grad_enabled(compute_gradients):
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                inputs.requires_grad_(compute_gradients and (grad_keys is None or set(grad_keys).issubset(set(range(inputs.shape[1])))))

                outputs = self.model(inputs)

                row: List[float] = []
                row_grads: List[torch.Tensor] | None = None

                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                    if compute_gradients:
                        if row_grads is None:
                            row_grads = []
                        grad = torch.autograd.grad(
                            outputs=scalar,
                            inputs=inputs,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad_keys is not None:
                            grad = grad[:, grad_keys]
                        row_grads.append(grad.detach().cpu())

                results.append(row)
                if compute_gradients:
                    if gradients is None:
                        gradients = []
                    gradients.append(row_grads)

        return results, gradients

    def add_noise(
        self,
        results: List[List[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Add Gaussian noise to a deterministic result set.

        Parameters
        ----------
        results : list of list of float
            Deterministic observable values.
        shots : int, optional
            Number of shots; if None, no noise is added.
        seed : int, optional
            Random seed for reproducibility.
        """
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["AdvancedBaseEstimator"]
