"""Enhanced lightweight estimator with batched evaluation and optional raw outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen:
    """Evaluate a PyTorch neural network for batches of inputs and observables.

    Parameters
    ----------
    model
        A ``torch.nn.Module`` that accepts a batch of parameter vectors and
        returns a tensor of shape ``(batch, output_dim)``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: torch.device | str = "cpu",
        batch_size: Optional[int] = None,
        return_raw: bool = False,
    ) -> List[List[float]]:
        """Compute the expectation of each observable for every parameter set.

        Parameters
        ----------
        observables
            Callables that map the network output to a scalar value.
        parameter_sets
            Sequence of parameter vectors.
        device
            Target device for the model and tensors.
        batch_size
            Process the parameters in chunks of this size; ``None`` means
            evaluate all at once.
        return_raw
            If ``True``, the raw model outputs are returned instead of
            observable values.

        Returns
        -------
        List[List[float]]
            A matrix of shape ``(len(parameter_sets), len(observables))`` or
            ``(len(parameter_sets), output_dim)`` when ``return_raw`` is True.
        """
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]

        results: List[List[float]] = []

        self.model.eval()
        device = torch.device(device)
        self.model.to(device)

        # Convert all parameters to a single tensor for efficient batching
        all_params = torch.stack([_ensure_batch(p) for p in parameter_sets]).squeeze(1).to(device)

        # Determine batch boundaries
        if batch_size is None or batch_size <= 0:
            batch_size = len(all_params)

        for start in range(0, len(all_params), batch_size):
            batch = all_params[start : start + batch_size]
            with torch.no_grad():
                outputs = self.model(batch)

            if return_raw:
                # Convert raw outputs to list of lists
                batch_results = outputs.cpu().tolist()
                results.extend(batch_results)
                continue

            # Compute observable values for the batch
            for i in range(batch.shape[0]):
                row: List[float] = []
                output = outputs[i]
                for observable in observables:
                    value = observable(output)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        return results


class FastEstimatorGen(FastBaseEstimatorGen):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> List[List[float]]:
        """Return noisy estimates by adding Gaussian noise with variance ``1/shots``.

        Any additional keyword arguments are forwarded to the base ``evaluate`` method.
        """
        raw = super().evaluate(observables, parameter_sets, **kwargs)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimatorGen", "FastEstimatorGen"]
