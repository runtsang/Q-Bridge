"""Enhanced neural‑network estimator with GPU support and gradient capabilities.

The class mirrors the original FastEstimator but adds:
* Optional device selection (CPU/GPU).
* Shot‑noise simulation via Gaussian perturbations.
* Analytic gradients of any scalar observable with respect to network parameters.
* Batch evaluation for efficiency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Neural‑network estimator with optional shot noise and gradient support."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str | None = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            A PyTorch ``nn.Module`` that maps input parameters to outputs.
        device
            Target device.  ``None`` defaults to ``'cpu'``; if a CUDA device is available
            it can be specified as ``'cuda'`` or ``torch.device``.
        shots
            Optional number of shots to simulate measurement noise.  If ``None`` the
            estimator is deterministic.
        seed
            Random seed for the noise generator.
        """
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate a list of scalar observables for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map a model output tensor to a scalar.
        parameter_sets
            Sequence of input parameter sequences (each should match model input size).

        Returns
        -------
        List[List[float]]
            Outer list indexes over parameter sets, inner list over observables.
        """
        obs = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params, self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in obs:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if self.shots is not None:
                    row = [
                        float(self._rng.normal(mean, max(1e-6, 1 / self.shots)))
                        for mean in row
                    ]

                results.append(row)
        return results

    def gradient_of(
        self,
        observable: ScalarObservable,
        parameter_set: Sequence[float],
    ) -> List[float]:
        """
        Compute the analytic gradient of a scalar observable w.r.t. model parameters.

        Parameters
        ----------
        observable
            Callable that maps model outputs to a scalar.
        parameter_set
            Input parameters for the forward pass.

        Returns
        -------
        List[float]
            Gradient vector flattened to a 1‑D list.
        """
        self.model.train()
        inputs = _ensure_batch(parameter_set, self.device)
        inputs.requires_grad_(True)

        outputs = self.model(inputs)
        value = observable(outputs)
        if isinstance(value, torch.Tensor):
            scalar = value.mean()
        else:
            scalar = torch.tensor(float(value), device=self.device, requires_grad=True)

        grads = grad(scalar, self.model.parameters(), create_graph=False)
        flat_grad = torch.cat([g.reshape(-1) for g in grads]).detach().cpu().numpy()
        return flat_grad.tolist()

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Vectorised evaluation that returns a NumPy array of shape
        ``(len(parameter_sets), len(observables))``.
        """
        return np.array(self.evaluate(observables, parameter_sets), dtype=np.float64)


__all__ = ["FastEstimator"]
