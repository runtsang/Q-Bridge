"""FastEstimator: a lightweight, GPU‑aware neural‑network estimator with support for batched evaluation, Gaussian shot noise, and automatic gradient computation.

The class wraps any :class:`torch.nn.Module` and exposes a flexible evaluate interface that accepts arbitrary
scalar observables (callables), multiple parameter sets, optional shot noise, and a target device.
Gradients of the observables with respect to the parameters are obtained via PyTorch autograd.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor
from typing import Callable, Iterable, List, Sequence, Dict, Tuple, Any, Optional

ScalarObservable = Callable[[Tensor], Tensor | float]
ObservableDict = Dict[str, ScalarObservable]

def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a 1‑D parameter array into a 2‑D batch tensor."""
    t = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

class FastEstimator:
    """Evaluate a PyTorch model over many parameter sets with optional shot noise and gradient support."""
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            A PyTorch module that maps a batch of parameters to outputs.
        device
            Target device for computation.  ``None`` uses ``torch.device('cpu')``.
        batch_size
            If provided, parameter sets are evaluated in mini‑batches of this size.
        """
        self.model = model
        self.device = torch.device(device or "cpu")
        self.batch_size = batch_size

    def _prepare_inputs(self, param_set: Sequence[Sequence[float]]) -> Tensor:
        """Batch and move parameters to the target device."""
        arr = torch.as_tensor(param_set, dtype=torch.float32, device=self.device)
        return arr

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | ObservableDict,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute (possibly noisy) expectation values for each parameter set.

        Parameters
        ----------
        observables
            Either a list of callables or a mapping from names to callables.
        parameter_sets
            Iterable of parameter vectors.  Each vector will be fed to the model.
        shots
            If provided, Gaussian shot noise with variance 1/shots is added to each mean.
        seed
            Random seed for shot noise.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        # Normalise observables to a list
        if isinstance(observables, dict):
            obs_list = list(observables.values())
        else:
            obs_list = list(observables) or [lambda out: out.mean(dim=-1)]

        # Evaluate in batches to keep memory usage bounded
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            if self.batch_size is None:
                batch = self._prepare_inputs(parameter_sets)
                outputs = self.model(batch)
                for params, out in zip(parameter_sets, outputs):
                    row = []
                    for obs in obs_list:
                        val = obs(out)
                        if isinstance(val, Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
            else:
                for start in range(0, len(parameter_sets), self.batch_size):
                    batch = self._prepare_inputs(parameter_sets[start:start + self.batch_size])
                    outputs = self.model(batch)
                    for out in outputs:
                        row = []
                        for obs in obs_list:
                            val = obs(out)
                            if isinstance(val, Tensor):
                                scalar = float(val.mean().cpu())
                            else:
                                scalar = float(val)
                            row.append(scalar)
                        results.append(row)

        # Add shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    def gradients(
        self,
        observables: Iterable[ScalarObservable] | ObservableDict,
        parameter_sets: Sequence[Sequence[float]],
        *,
        retain_graph: bool = False,
    ) -> List[List[Tensor]]:
        """
        Compute gradients of each observable w.r.t. the model parameters.

        Parameters
        ----------
        observables
            List or dict of observable callables.
        parameter_sets
            Iterable of parameter vectors.
        retain_graph
            Whether to keep the computation graph for chained gradients.

        Returns
        -------
        List[List[Tensor]]
            Outer list over parameter sets, inner list over observables.
            Each tensor has shape matching the model parameters.
        """
        if isinstance(observables, dict):
            obs_list = list(observables.values())
        else:
            obs_list = list(observables) or [lambda out: out.mean(dim=-1)]

        grads: List[List[Tensor]] = []
        self.model.train()  # enable gradient tracking
        for params in parameter_sets:
            param_tensor = _ensure_batch(params).requires_grad_(True)
            outputs = self.model(param_tensor)
            row: List[Tensor] = []
            for obs in obs_list:
                val = obs(outputs)
                if isinstance(val, Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, dtype=outputs.dtype, device=outputs.device)
                val.backward(retain_graph=retain_graph)
                row.append(param_tensor.grad.clone())
                param_tensor.grad.zero_()
            grads.append(row)
        return grads

__all__ = ["FastEstimator"]
