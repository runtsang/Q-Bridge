"""Enhanced lightweight estimator utilities for classical neural networks.

Features:
- Batch evaluation of arbitrary scalar observables.
- Optional GPU acceleration.
- Shot‑noise simulation via Gaussian perturbations.
- Automatic gradient computation of expectation values w.r.t. model parameters.
- Flexible device and loss‑function support.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor

ScalarObservable = Callable[[Tensor], Tensor | float]
Device = Union[str, torch.device]

def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a sequence of floats to a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for a set of parameter vectors.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  The model must accept a batch of
        parameter vectors and return a tensor of outputs.
    device : str | torch.device, optional
        The device on which to run the model.  Defaults to ``'cpu'``.
    """

    def __init__(self, model: nn.Module, device: Device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def _evaluate_batch(
        self,
        observables: List[ScalarObservable],
        parameter_sets: Tensor,
    ) -> Tensor:
        """Internal helper that runs the model once and applies all observables."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(parameter_sets.to(self.device))
        # ``outputs`` shape: (batch, out_dim)
        # Apply each observable to the outputs.
        results = torch.stack([self._apply_observable(obs, outputs) for obs in observables], dim=1)
        return results.cpu()

    @staticmethod
    def _apply_observable(obs: ScalarObservable, outputs: Tensor) -> Tensor:
        """Apply an observable callable to the model outputs."""
        val = obs(outputs)
        if isinstance(val, Tensor):
            return val.mean(dim=-1)
        return torch.tensor(val, dtype=torch.float32)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a 2‑D list of scalar expectation values.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to evaluate.
        """
        obs_list = list(observables) or [lambda out: out.mean(dim=-1)]
        batch = _ensure_batch(parameter_sets)
        results = self._evaluate_batch(obs_list, batch)
        return results.numpy().tolist()

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Same as :meth:`evaluate` but adds Gaussian shot noise."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in raw]
        return noisy

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        retain_graph: bool = False,
    ) -> List[List[List[float]]]:
        """Compute the gradient of each observable w.r.t. each parameter.

        Returns a list of shape (num_sets, num_observables, num_params).
        """
        obs_list = list(observables) or [lambda out: out.mean(dim=-1)]
        batch = _ensure_batch(parameter_sets)
        batch.requires_grad_(True)

        self.model.train()
        outputs = self.model(batch.to(self.device))
        grads = []
        for obs in obs_list:
            val = self._apply_observable(obs, outputs)
            grad = torch.autograd.grad(
                outputs=val,
                inputs=batch,
                grad_outputs=torch.ones_like(val),
                retain_graph=retain_graph,
                create_graph=False,
            )[0]
            grads.append(grad.cpu().numpy())
        # grads shape: (num_obs, batch, params)
        grads_np = np.stack(grads, axis=0)
        # transpose to (batch, num_obs, params)
        grads_np = np.transpose(grads_np, (1, 0, 2))
        return grads_np.tolist()

__all__ = ["FastBaseEstimator"]
