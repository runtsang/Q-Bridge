"""Hybrid estimator for classical PyTorch models, providing deterministic
evaluation, optional Gaussian shot‑noise, and helpers for LSTM
sequence tagging and a toy fully‑connected layer."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Tuple, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Turn a 1‑D list of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FCL:
    """Plain‑Python stand‑in for a fully‑connected quantum layer."""
    def __init__(self, n_features: int = 1) -> None:
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()


class HybridEstimator:
    """Evaluate a PyTorch model for parameter sets with observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    # --------------------------- core evaluation ---------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set using the supplied
        observables.  If ``shots`` is given, Gaussian noise simulating
        quantum shot‑noise is added to the deterministic results.
        """
        raw = self._evaluate_deterministic(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _evaluate_deterministic(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    # --------------------------- helpers -----------------------------------
    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the toy fully‑connected layer."""
        fcl = FCL()
        return fcl.run(thetas)

    def evaluate_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper for an ``LSTMTagger`` model.  Returns the
        log‑softmax of tag logits.
        """
        if not hasattr(self.model, "lstm"):
            raise TypeError("Model does not contain an LSTMTagger.")
        embeds = self.model.word_embeddings(sentence)
        lstm_out, _ = self.model.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.model.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)


__all__ = ["HybridEstimator", "FCL"]
