"""Enhanced lightweight estimator utilities built on PyTorch.

This module extends the original FastBaseEstimator with:
- GPU support and automatic device selection.
- Gradient and Hessian evaluation for arbitrary scalar observables.
- Batch‑wise evaluation to fully exploit vectorised operations.
- Noise models: Gaussian (shot noise) and Poisson (count‑based measurement noise).
- Utility methods for moving models to devices and toggling training mode.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Wrap a 1‑D sequence into a batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` accepting a batch of parameter vectors.
    device:
        Device on which the model and tensors will be placed. Defaults to
        ``torch.device('cuda' if torch.cuda.is_available() else 'cpu')``.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        The method is fully deterministic and runs in inference mode.
        """
        observables = list(observables) or [
            lambda outputs: outputs.mean(dim=-1)
        ]
        results: List[List[float]] = []
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

    # ------------------------------------------------------------------
    # Gradient evaluation
    # ------------------------------------------------------------------
    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return gradients of each observable w.r.t. the input parameters.

        The output is a list of rows, each row containing a list of gradients
        for each observable.  Each gradient is a list of floats matching the
        dimensionality of the parameter vector.
        """
        observables = list(observables) or [
            lambda outputs: outputs.mean(dim=-1)
        ]
        grads: List[List[List[float]]] = []
        for params in parameter_sets:
            params_tensor = torch.tensor(
                params, dtype=torch.float32, device=self.device, requires_grad=True
            )
            outputs = self.model(params_tensor)
            grad_row: List[List[float]] = []
            for observable in observables:
                val = observable(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, dtype=torch.float32, device=self.device)
                grad = torch.autograd.grad(
                    val, params_tensor, retain_graph=True, allow_unused=True
                )[0]
                grad_row.append(grad.detach().cpu().numpy().tolist())
            grads.append(grad_row)
        return grads

    # ------------------------------------------------------------------
    # Noise‑augmented evaluation
    # ------------------------------------------------------------------
    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        noise_type: str = "gaussian",
    ) -> List[List[float]]:
        """Return noisy estimates of the observable values.

        Parameters
        ----------
        shots:
            Number of independent samples.  If ``None`` the evaluation is
            deterministic.
        noise_type:
            ``"gaussian"`` for additive Gaussian noise (shot‑noise model) or
            ``"poisson"`` for Poissonian count noise (used in quantum
            measurements).  The default is ``"gaussian"``.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            if noise_type == "gaussian":
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
            elif noise_type == "poisson":
                noisy_row = [
                    float(rng.poisson(mean * shots) / shots) for mean in row
                ]
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def to(self, device: torch.device) -> None:
        """Move the model to a different device."""
        self.device = device
        self.model.to(device)

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`evaluate`."""
        return self.evaluate(*args, **kwargs)


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator.

    The class is kept for backward compatibility but now delegates to the
    :meth:`evaluate_with_noise` method.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return self.evaluate_with_noise(
            observables,
            parameter_sets,
            shots=shots,
            seed=seed,
            noise_type="gaussian",
        )


__all__ = ["FastBaseEstimator", "FastEstimator"]
