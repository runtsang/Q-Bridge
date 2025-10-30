"""
Hybrid FastBaseEstimator for classical models with optional self‑attention and shot noise.

This module extends the original FastBaseEstimator by adding:
- A plug‑in classical self‑attention block (from SelfAttention.py).
- Support for Gaussian shot noise in the same API as the quantum version.
- A unified evaluate signature that works for both deterministic and noisy runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# Import the classical self‑attention helper
try:
    from.SelfAttention import SelfAttention as ClassicalSelfAttention
except Exception:  # pragma: no cover
    ClassicalSelfAttention = None

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of inputs with optional self‑attention.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    self_attention : bool, default False
        Enable a classical self‑attention preprocessing step.
    attention_embed_dim : int, optional
        The embedding dimension for the self‑attention block.  Required if
        ``self_attention`` is True.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        self_attention: bool = False,
        attention_embed_dim: Optional[int] = None,
    ) -> None:
        self.model = model
        self.self_attention = self_attention
        if self_attention:
            if attention_embed_dim is None:
                raise ValueError("attention_embed_dim must be specified when self_attention is True")
            self._attention = ClassicalSelfAttention(attention_embed_dim)  # type: ignore
        else:
            self._attention = None

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply self‑attention if enabled."""
        if self.self_attention and self._attention is not None:
            # Convert tensors to numpy for the external helper
            inp_np = inputs.detach().cpu().numpy()
            # For demo purposes we use identity rotation and entangle params
            rot = np.eye(self._attention.embed_dim).flatten()
            ent = np.eye(self._attention.embed_dim).flatten()
            out = self._attention.run(rot, ent, inp_np)
            return torch.as_tensor(out, dtype=inputs.dtype, device=inputs.device)
        return inputs

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map model outputs to scalars.  If empty a
            default mean‑over‑features observable is used.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed into the model.
        shots : int, optional
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                inputs = self._preprocess(inputs)
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


__all__ = ["FastBaseEstimator"]
