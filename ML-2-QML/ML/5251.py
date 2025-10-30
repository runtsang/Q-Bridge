from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import numpy as np

# Import anchor circuit factory
from.QuantumClassifierModel import build_classifier_circuit as build_cls_circuit
# Efficient estimator utilities
from.FastBaseEstimator import FastEstimator
# Classical kernel
from.QuantumKernelMethod import Kernel

class HybridClassifier:
    """
    Classical hybrid classifier that mirrors the quantum helper interface.

    * Builds a deep feed‑forward network via :func:`build_cls_circuit`.
    * Uses :class:`FastEstimator` for batched, optional shot‑noisy inference.
    * Provides an RBF kernel matrix through :class:`Kernel` for downstream methods
      such as kernel‑based SVMs or Gaussian processes.
    """

    def __init__(self, num_features: int, depth: int, gamma: float = 1.0,
                 shots: int | None = None, seed: int | None = None) -> None:
        self.net, self.encoding, self.weight_sizes, self.observables = \
            build_cls_circuit(num_features, depth)
        self.estimator = FastEstimator(self.net)
        self.shots = shots
        self.seed = seed
        self.kernel = Kernel(gamma)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for the given batch."""
        logits = self.net(X)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def evaluate(self, X: torch.Tensor, observables: Iterable[torch.Tensor] | None = None) -> List[List[float]]:
        """Fast batched evaluation with optional Gaussian shot noise."""
        return self.estimator.evaluate(
            observables or [lambda out: out.mean(dim=-1)],
            X.tolist(),
            shots=self.shots,
            seed=self.seed
        )

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the classical RBF kernel."""
        return np.array([[self.kernel(x, y).item() for y in Y] for x in X])

    def fit(self, *_, **__):
        """Placeholder – training logic can be added as required."""
        raise NotImplementedError("Training is not implemented in this skeleton.")
