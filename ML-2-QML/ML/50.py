"""Vectorized neural‑network estimator with optional noise and GPU support."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Optional, Union

ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float, np.ndarray]]


class FastBaseEstimator:
    """Evaluate a PyTorch model over many parameter sets with optional noise.

    Parameters
    ----------
    model:
        Any nn.Module that maps a 1‑D input tensor to an output tensor.
    device:
        Device to run the model on. Defaults to CUDA if available.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prepare_inputs(self, parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        tensor = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
        dropout: float | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Callables that operate on the model output tensor and return a scalar
            or a tensor with the same batch shape. If None a mean over the last
            dimension is used.
        parameter_sets:
            Iterable of parameter sequences. If None an empty list is returned.
        shots:
            If set, Gaussian noise with std = 1/sqrt(shots) is added to each result.
        seed:
            RNG seed for reproducibility when shots is set.
        dropout:
            Dropout probability applied to the model output before evaluating
            observables. Useful for simulating measurement noise.
        """
        if parameter_sets is None:
            return []

        inputs = self._prepare_inputs(parameter_sets)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)

        if dropout is not None:
            outputs = nn.Dropout(p=dropout)(outputs)

        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[np.ndarray] = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            else:
                val = np.array(val)
            results.append(val)

        results = np.stack(results, axis=0)  # shape (obs, batch)

        if shots is not None:
            rng = np.random.default_rng(seed)
            std = np.maximum(1e-6, 1.0 / np.sqrt(shots))
            noise = rng.normal(0.0, std, size=results.shape)
            results += noise

        return results.T.tolist()


__all__ = ["FastBaseEstimator"]
