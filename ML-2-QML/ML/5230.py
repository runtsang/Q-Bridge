"""Hybrid fast estimator combining classical PyTorch and quantum‑like callables.

The estimator can evaluate:
* Pure PyTorch modules.
* Callable objects that emulate a quantum circuit and return NumPy arrays.
* Sequence tagging models (e.g. LSTMTagger).
* Graph neural networks implemented with simple fully‑connected layers.

Shot noise may be injected post‑hoc.  The module also exposes a helper
``FCL`` that mimics a fully‑connected layer using a classical
neural network.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
QuantumObservable = Callable[[np.ndarray], float]
ModelType = Union[nn.Module, Callable[[Sequence[float]], np.ndarray], Any]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Lightweight estimator that works with both classical and quantum‑like models."""

    def __init__(self, model: ModelType) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Generic evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, QuantumObservable]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of observables for each parameter set.

        * For a PyTorch model ``observables`` must be callables that take a
          ``torch.Tensor`` and return a scalar (or a tensor that can be
          reduced to a scalar).
        * For a quantum‑like callable ``observables`` must accept a NumPy
          array and return a float.
        """
        if not observables:
            raise ValueError("At least one observable must be provided")

        results: List[List[float]] = []

        is_torch = isinstance(self.model, nn.Module)

        for params in parameter_sets:
            if is_torch:
                inputs = _ensure_batch(params)
                with torch.no_grad():
                    outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
            else:
                # Assume a callable that returns a NumPy array of expectations
                raw = self.model(params)
                if not isinstance(raw, np.ndarray):
                    raise TypeError("Quantum model must return a NumPy array")
                row = [float(obs(raw)) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    # ------------------------------------------------------------------
    # Sequence evaluation (for LSTMTagger / QLSTMTagger)
    # ------------------------------------------------------------------
    def evaluate_sequence(
        self,
        sentence: torch.Tensor,
        tagset_size: int,
    ) -> torch.Tensor:
        """
        Convenience wrapper for models that expose a ``forward`` method
        returning log‑softmax tag logits (e.g. ``LSTMTagger``).
        """
        if not hasattr(self.model, "forward"):
            raise AttributeError("Model does not support sequence evaluation")
        if not isinstance(self.model, nn.Module):
            raise TypeError("Sequence evaluation requires a PyTorch module")
        embeds = self.model.word_embeddings(sentence)
        lstm_out, _ = self.model.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.model.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

    # ------------------------------------------------------------------
    # Graph neural network evaluation
    # ------------------------------------------------------------------
    def evaluate_graph(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """
        Evaluate a graph neural network implemented with simple
        fully‑connected layers.  ``samples`` should be an iterable of
        (feature, target) pairs.  The method returns a list of lists
        containing the activations for each layer.
        """
        if not hasattr(self.model, "weights"):
            raise AttributeError("Model does not provide 'weights' attribute")
        activations: List[List[np.ndarray]] = []
        for features, _ in samples:
            current = torch.from_numpy(features).float()
            layer_acts = [current.numpy()]
            for w in self.model.weights:
                current = torch.tanh(torch.matmul(w, current))
                layer_acts.append(current.numpy())
            activations.append(layer_acts)
        return activations


# ----------------------------------------------------------------------
# Helper: fully connected layer (classical)
# ----------------------------------------------------------------------
class FCL:
    """Return an object with a ``run`` method mimicking the quantum example."""

    def __init__(self, n_features: int = 1) -> None:
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


__all__ = ["HybridFastEstimator", "FCL"]
