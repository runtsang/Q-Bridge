"""Hybrid QRNN implementation that can run in classical mode.

This module builds a classical neural network using the same layer
construction logic as the quantum variant, evaluates it with
`FastBaseEstimator`, and provides a minimal gradient‑descent training loop.
It imports utilities from the original QRNN and QuantumClassifierModel
packages but does not duplicate any code verbatim."""
from __future__ import annotations

import torch
from torch import nn, Tensor
from typing import List, Sequence, Tuple

# Utilities from the original QRNN (random unitary generation, etc.)
from QRNN import random_unitary, random_state, random_unitaries, feedforward, cost
# Classical classifier factory
from QuantumClassifierModel import build_classifier_circuit
# Estimator primitives
from FastBaseEstimator import FastBaseEstimator, FastEstimator


class HybridQRNN:
    """Hybrid QRNN that can operate in classical or quantum mode."""
    def __init__(self, num_features: int, depth: int, use_quantum: bool = False):
        self.num_features = num_features
        self.depth = depth
        self.use_quantum = use_quantum

        if use_quantum:
            # Quantum branch – build a parameterised Qiskit circuit
            self.circuit, self.enc_params, self.weight_params, self.observables = build_classifier_circuit(
                num_features, depth
            )
            self.estimator = FastEstimator(self.circuit)
        else:
            # Classical branch – build a PyTorch sequential model
            self.network, self.enc_params, self.weight_sizes, self.observables = build_classifier_circuit(
                num_features, depth
            )
            self.estimator = FastBaseEstimator(self.network)

        # Flatten all trainable parameters into a list for easy manipulation
        self.params = self._init_params()

    def _init_params(self) -> List[float]:
        if self.use_quantum:
            return [float(p) for p in self.enc_params + self.weight_params]
        else:
            return [float(p) for p in self.network.parameters()]

    def evaluate(self, param_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate a batch of parameter sets against the stored observables."""
        return self.estimator.evaluate(self.observables, param_sets)

    def train(
        self,
        training_set: Sequence[Tuple[Tensor, Tensor]],
        lr: float = 0.01,
        epochs: int = 10,
    ) -> None:
        """Simple SGD training loop for the classical branch."""
        if self.use_quantum:
            raise NotImplementedError("Quantum training not provided in this module.")
        optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for x, y in training_set:
                optimizer.zero_grad()
                out = self.network(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

    def get_parameters(self) -> List[float]:
        """Return current parameters as a flat list."""
        return self._init_params()


__all__ = ["HybridQRNN"]
