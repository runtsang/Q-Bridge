"""Extended estimator utilities implemented with PyTorch modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Type definitions
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a batched torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Features
    --------
    * Optional GPU/CPU device selection.
    * Automatic gradient computation for any observable that depends on the
      network output.
    * Batch‑wise evaluation for speed.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    # --------------------------------------------------------------------- #
    # Basic evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map a network output tensor to a scalar
            (or tensor that can be reduced to a scalar). If empty, the mean of
            the last dimension is used.
        parameter_sets:
            Sequence of parameter vectors (each a list/tuple of floats).
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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
        return results

    # --------------------------------------------------------------------- #
    # Gradient‑aware evaluation
    # --------------------------------------------------------------------- #
    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[List[torch.Tensor]]]]:
        """Return values and gradients for each observable.

        The returned gradients are tensors that share the same shape as the
        model parameters. They are detached from the computation graph and
        placed on CPU.

        Parameters
        ----------
        observables:
            Iterable of callable observables.
        parameter_sets:
            Sequence of parameter vectors.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        values: List[List[float]] = []
        grads: List[List[List[torch.Tensor]]] = []

        self.model.eval()
        for params in parameter_sets:
            # Enable gradient tracking on model parameters
            for p in self.model.parameters():
                p.requires_grad_(True)

            inputs = _ensure_batch(params).to(self.device)
            outputs = self.model(inputs)

            val_row: List[float] = []
            grad_row: List[List[torch.Tensor]] = []

            for observable in observables:
                # Compute the scalar output
                scalar = observable(outputs)
                if isinstance(scalar, torch.Tensor):
                    scalar = scalar.mean()

                # Backward pass to obtain gradients
                self.model.zero_grad()
                scalar.backward(retain_graph=True)

                # Collect gradients
                grad_params = [
                    p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p)
                    for p in self.model.parameters()
                ]

                # Scalar value
                val_row.append(float(scalar.item()))

                # Store gradient tensors per observable
                grad_row.append(grad_params)

            values.append(val_row)
            grads.append(grad_row)

        return values, grads


__all__ = ["FastBaseEstimator"]
