"""
Hybrid estimator that fuses a PyTorch neural network with optional classical
self‑attention preprocessing.  The API mirrors the original FastBaseEstimator
and adds a noise‑augmented wrapper.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional
import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
def ClassicalSelfAttention(embed_dim: int = 4) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a functional self‑attention block that operates on NumPy arrays.
    The implementation mirrors the original SelfAttention.py but is exposed
    as a stateless callable for easy composition.
    """
    def _self_attention(inputs: np.ndarray) -> np.ndarray:
        """
        Compute a simple scaled‑dot‑product self‑attention.

        Parameters
        ----------
        inputs : np.ndarray
            Input matrix of shape (batch, feature_dim).
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        # Random projection matrices – in a real model these would be learned.
        rot = torch.randn(inputs.shape[1], embed_dim, dtype=torch.float32)
        ent = torch.randn(inputs.shape[1], embed_dim, dtype=torch.float32)

        query = inputs_t @ rot
        key   = inputs_t @ ent
        value = inputs_t

        scores = torch.softmax(query @ key.T / np.sqrt(embed_dim), dim=-1)
        return (scores @ value).numpy()

    return _self_attention


# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridFastEstimator:
    """
    Evaluate a PyTorch model with optional self‑attention preprocessing
    and optional Gaussian shot noise.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model that accepts a batch of inputs and returns a batch of
        outputs.  The model must be in evaluation mode when passed to
        :meth:`evaluate`.
    attention : Callable[[np.ndarray], np.ndarray] | None
        A stateless self‑attention function applied to the raw parameters
        before feeding them into the model.  If ``None`` the raw parameters
        are used unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        attention: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.model = model
        self.attention = attention

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute observable values for each parameter set.

        Parameters
        ----------
        observables : iterable
            Callables that map a model output to a scalar (or tensor).
        parameter_sets : sequence
            Iterable of parameter vectors to evaluate.
        shots : int | None
            If provided, Gaussian noise with variance 1/shots is added to each
            mean value.
        seed : int | None
            Random seed for the noise generator.

        Returns
        -------
        List[List[float]]
            A list of rows, each row containing the scalar results for one
            parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        raw_results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Optional self‑attention preprocessing
                if self.attention is not None:
                    params = self.attention(np.array(params)).tolist()

                inputs = self._ensure_batch(params)
                outputs = self.model(inputs)

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                raw_results.append(row)

        if shots is None:
            return raw_results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw_results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridFastEstimator"]
