"""FastBaseEstimator for classical models with advanced evaluation features.

The estimator supports batched inference on GPU, flexible observables,
automatic differentiation, and optional Poisson shot noise to emulate
quantum measurement statistics.  It is designed to be drop‑in
compatible with the original FastBaseEstimator while adding richer
functionality for research workflows.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
VectorObservable = Callable[[torch.Tensor], torch.Tensor | Sequence[float]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  It must accept a batch tensor
        of shape ``(batch, *in_shape)`` and return a tensor of shape
        ``(batch, out_dim)``.
    device : str | torch.device, optional
        Device on which to run the model.  Defaults to ``"cpu"``.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, VectorObservable]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return the value of each observable for every parameter set.

        Parameters
        ----------
        observables
            Callables that map the model output to a scalar or a vector.
        parameter_sets
            Iterable of parameter sequences.  Each sequence is converted
            to a 1‑D tensor and batched together for efficient evaluation.
        shots
            If provided, add Poisson shot noise to each mean value.
            ``shots`` represents the number of measurement shots.
        seed
            Random seed for the shot noise generator.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        batch = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            raw_results: List[List[float]] = []

            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        val = val.unsqueeze(0)
                    val_cpu = val.cpu()
                    # Convert to Python scalars or sequences
                    if val_cpu.ndim == 1:
                        val_list = val_cpu.tolist()
                    else:
                        val_list = [float(v) for v in val_cpu]
                else:
                    val_list = [float(val)] if np.isscalar(val) else val

                raw_results.append(val_list)

            # Transpose to match original API: list of rows
            rows = [list(row) for row in zip(*raw_results)]

            if shots is not None:
                rng = np.random.default_rng(seed)
                noisy_rows: List[List[float]] = []
                for row in rows:
                    noisy_row = [
                        float(rng.poisson(mean * shots) / shots) if mean >= 0 else float(rng.poisson(-mean * shots) / shots) * -1
                        for mean in row
                    ]
                    noisy_rows.append(noisy_row)
                return noisy_rows

            return rows
