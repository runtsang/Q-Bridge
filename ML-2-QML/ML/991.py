"""Enhanced FastBaseEstimator leveraging PyTorch autograd and GPU acceleration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Dict, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional GPU support.

    Features
    --------
    * Automatic device placement (CPU or CUDA).
    * Batch evaluation of multiple parameter sets.
    * Optional Gaussian shot noise to mimic stochastic measurements.
    * Gradient estimation via PyTorch autograd for differentiable observables.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device or "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        return_std: bool = False,
    ) -> List[Dict[str, Any]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar
            (tensor or python float). If ``None`` a single mean output is used.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one forward pass.
        return_std : bool
            If True, also return the standard deviation of the observable
            over the batch of parameters.

        Returns
        -------
        List[Dict[str, Any]]
            For each parameter set a dict with keys ``values`` (list of floats)
            and optionally ``std`` (list of floats).
        """
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        if parameter_sets is None:
            parameter_sets = []

        results: List[Dict[str, Any]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params, self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                std_row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                        if return_std:
                            std = val / max(1e-6, np.sqrt(len(outputs)))
                            std_row.append(std)
                    else:
                        val = float(val)
                        if return_std:
                            std_row.append(0.0)
                    row.append(val)
                result: Dict[str, Any] = {"values": row}
                if return_std:
                    result["std"] = std_row
                results.append(result)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Add Gaussian shot noise to the deterministic estimates.

        Parameters
        ----------
        shots : int, optional
            Number of shots; if ``None`` no noise is added.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy_results: List[Dict[str, Any]] = []
        for res in raw:
            noisy_values = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in res["values"]
            ]
            noisy_results.append({"values": noisy_values})
        return noisy_results

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Compute gradients of observables w.r.t. model parameters via autograd.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict contains ``values`` (list of floats) and ``gradients`` (list of tensors)
            corresponding to the parameters of the model.
        """
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        if parameter_sets is None:
            parameter_sets = []

        results: List[Dict[str, Any]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params, self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            row: List[float] = []
            grads: List[torch.Tensor] = []

            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, device=self.device)
                row.append(val.item())

                # compute gradient of this observable w.r.t. inputs
                self.model.zero_grad()
                val.backward(retain_graph=True)
                grads.append(inputs.grad.clone().detach().cpu())
                inputs.grad.zero_()

            results.append({"values": row, "gradients": grads})
        return results


__all__ = ["FastBaseEstimator"]
