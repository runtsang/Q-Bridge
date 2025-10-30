"""Hybrid estimator that combines a PyTorch neural network with optional attention and noise modelling."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator(nn.Module):
    """A PyTorch module that accepts a parameter vector and produces a feature vector.

    The network can be built from a simple dense layer or a lightweight attention
    mechanism that uses *all* parameter‑values as input.  The module inherits
    from ``nn.Module`` so that it can be trainable via standard PyTorch
    optimisers.

    Parameters
    ----------
    model
        A user‑supplied ``nn.Module`` that maps a tensor of shape ``(batch,
        n_params)`` to an output tensor.  If ``None`` a small MLP is built
        automatically.
    use_attention
        If ``True`` an attention layer is inserted before the final linear
        output.  The attention weights are computed from a linear
        transformation of the input parameters and normalised with a softmax.
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        *,
        hidden_sizes: tuple[int,...] | None = None,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        if model is not None:
            self.model = model
        else:
            # Build a simple MLP when no model is supplied
            layers: List[nn.Module] = []
            if hidden_sizes is None:
                layers.append(nn.Linear(1, 1))
            else:
                in_features = 1
                for size in hidden_sizes:
                    layers.append(nn.Linear(in_features, size))
                    layers.append(nn.ReLU())
                    in_features = size
                layers.append(nn.Linear(in_features, 1))
            self.model = nn.Sequential(*layers)

        if self.use_attention:
            # The attention layer will be initialised lazily during the first
            # forward pass once the output dimensionality is known.
            self._attention: Optional[nn.Linear] = None

    def _init_attention(self, out_features: int) -> None:
        self._attention = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the feature vector for a batch of parameter vectors."""
        out = self.model(x)
        if self.use_attention:
            if self._attention is None:
                self._init_attention(out.shape[-1])
            attn_weights = F.softmax(self._attention(out), dim=-1)
            out = out * attn_weights
        return out

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a list of observables for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
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

    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> np.ndarray:
        """Convenience method that returns the raw network output."""
        self.eval()
        with torch.no_grad():
            inputs = torch.as_tensor(parameter_sets, dtype=torch.float32)
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            outputs = self(inputs)
            return outputs.cpu().numpy()

    def _add_noise(
        self,
        data: List[List[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to the deterministic estimator output."""
        if shots is None:
            return data
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in data:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate with optional shot noise."""
        raw = self.evaluate(observables, parameter_sets)
        return self._add_noise(raw, shots, seed)


__all__ = ["FastHybridEstimator"]
