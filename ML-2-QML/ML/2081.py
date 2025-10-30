from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Fast evaluation of PyTorch models with optional GPU support and analytic gradients.

    Parameters
    ----------
    model : nn.Module
        Neural network that maps input tensors to outputs.
    device : str | torch.device | None, optional
        Target device for model and computations (default: ``'cpu'``).
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Evaluate a list of observables on the model for many parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Each callable maps the model output tensor to a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence is a vector of input parameters.

        Returns
        -------
        np.ndarray
            Shape ``(N, M)`` where *N* is the number of parameter sets
            and *M* is the number of observables.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        float_val = float(val.mean().cpu())
                    else:
                        float_val = float(val)
                    row.append(float_val)
                results.append(row)

        return np.array(results, dtype=np.float64)

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        """
        Evaluate observables and their gradients w.r.t the input parameters.

        Returns
        -------
        expectations : np.ndarray
            Array of shape ``(N, M)`` with expectation values.
        gradients : list of list of np.ndarray
            For each parameter set and each observable a gradient vector
            of shape ``(D,)`` where *D* is the dimensionality of the input.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        expectations: List[List[float]] = []
        gradients: List[List[np.ndarray]] = []

        self.model.eval()
        for params in parameter_sets:
            input_tensor = _ensure_batch(params).to(self.device)
            input_tensor.requires_grad_(True)
            outputs = self.model(input_tensor)
            row_exp: List[float] = []
            row_grad: List[np.ndarray] = []

            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, dtype=torch.float32, device=self.device)

                row_exp.append(float(val.cpu()))
                grad = torch.autograd.grad(val, input_tensor, retain_graph=True)[0]
                row_grad.append(grad.squeeze().detach().cpu().numpy())

            expectations.append(row_exp)
            gradients.append(row_grad)

        return np.array(expectations, dtype=np.float64), gradients


__all__ = ["FastBaseEstimator"]
