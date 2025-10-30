"""Hybrid estimator combining PyTorch models and classical self‑attention."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention:
    """Lightweight self‑attention block that mimics the quantum interface."""

    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class FastHybridEstimator:
    """
    Evaluate either a PyTorch neural network or a classical self‑attention
    block.  The API mirrors the original FastBaseEstimator, but the
    constructor accepts any callable with a ``run`` or ``forward`` method.
    """

    def __init__(self, model: Union[nn.Module, Callable[..., np.ndarray]]) -> None:
        self.model = model
        if isinstance(model, nn.Module):
            self._is_nn = True
        else:
            # Assume a callable with a run method (e.g., ClassicalSelfAttention)
            self._is_nn = False

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute outputs for each parameter set.  For neural nets the
        ``observables`` are applied to the network output.  For a
        self‑attention block the ``observables`` are ignored and the raw
        attention output is returned.
        """
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        if not self._is_nn:
            # Self‑attention block: expectation value is the raw output
            results: List[List[float]] = []
            for params in parameter_sets:
                # params expected: [rot_params..., ent_params..., input_vector]
                rot_len = len(params) // 3
                rot_params = np.array(params[:rot_len])
                ent_params = np.array(params[rot_len:2 * rot_len])
                input_vec = np.array(params[2 * rot_len:])
                out = self.model.run(rot_params, ent_params, input_vec)
                results.append([float(v) for v in out])
            return results

        # Neural network path
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "ClassicalSelfAttention"]
