"""Enhanced estimator for PyTorch neural networks with batched inference,
gradient support, and optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Union[Sequence[float], torch.Tensor]) -> torch.Tensor:
    """Convert a sequence or tensor to a 2‑D batch tensor on the default device."""
    if isinstance(values, torch.Tensor):
        tensor = values.to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  The model is moved to *device*.
    device : str | torch.device, optional
        Device on which to run the model.  Defaults to ``"cpu"``.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor,
        *,
        gradient: bool = False,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        If *gradient* is True, the model is evaluated in autograd mode and each
        observable is expected to be a differentiable function of the outputs.
        In that case the returned values are the gradients of the observable
        with respect to the model parameters, flattened into a list.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Functions that map the model output to a scalar (or list of scalars).
        parameter_sets : Sequence[Sequence[float]] | torch.Tensor
            Batch of parameter vectors for which to evaluate the model.
        gradient : bool, optional
            Whether to compute gradients of observables w.r.t. model parameters.
        """
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        observables = list(observables)

        batch = _ensure_batch(parameter_sets).to(self.device)
        self.model.eval()
        results: List[List[float]] = []

        if gradient:
            # Enable autograd for each parameter set
            for params in batch:
                params = params.clone().requires_grad_(True)
                outputs = self.model(params)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        grad = torch.autograd.grad(
                            val.sum(), self.model.parameters(), retain_graph=True
                        )
                        grads = torch.cat([g.flatten() for g in grad if g is not None])
                        row.append(float(grads.mean().item()))
                    else:
                        row.append(float(val))
                results.append(row)
        else:
            with torch.no_grad():
                outputs = self.model(batch)
                for obs in observables:
                    out = obs(outputs)
                    if isinstance(out, torch.Tensor):
                        out = out.cpu()
                        if out.ndim == 0:
                            out = out.unsqueeze(0)
                        results.append(out.numpy().tolist())
                # Transpose to match original shape
                results = list(map(list, zip(*results)))

        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Same as :meth:`evaluate` but adds Gaussian shot noise.

        Parameters
        ----------
        shots : int | None
            Number of shots; if ``None`` no noise is applied.
        seed : int | None
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets, gradient=False)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
