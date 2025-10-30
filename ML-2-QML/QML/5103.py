"""Hybrid QRNN implementation that can run in quantum mode.

This module mirrors the classical API but builds a Qiskit circuit,
binds parameters, evaluates using a FastBaseEstimator, and provides
basic utilities for parameter extraction.  It imports the same
construction logic as the quantum classifier and reuses the
estimator primitives from the original QML seed."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Utilities from the original QRNN (random unitary generation, etc.)
from QRNN import random_unitary
# Quantum classifier factory
from QuantumClassifierModel import build_classifier_circuit
# Estimator primitives
from FastBaseEstimator import FastBaseEstimator


class HybridQRNN:
    """Hybrid QRNN that can operate in classical or quantum mode."""
    def __init__(self, num_qubits: int, depth: int, use_quantum: bool = True):
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_quantum = use_quantum

        if use_quantum:
            # Quantum branch – build a parameterised Qiskit circuit
            self.circuit, self.enc_params, self.weight_params, self.observables = build_classifier_circuit(
                num_qubits, depth
            )
            self.estimator = FastBaseEstimator(self.circuit)
        else:
            # Classical branch – build a PyTorch sequential model
            from torch import nn
            self.network, self.enc_params, self.weight_sizes, self.observables = build_classifier_circuit(
                num_qubits, depth
            )
            self.estimator = FastBaseEstimator(self.network)

        self.params = self._init_params()

    def _init_params(self) -> List[float]:
        if self.use_quantum:
            return [float(p) for p in self.enc_params + self.weight_params]
        else:
            return [float(p) for p in self.network.parameters()]

    def evaluate(self, param_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Evaluate a batch of parameter sets against the stored observables."""
        return self.estimator.evaluate(self.observables, param_sets)

    def get_parameters(self) -> List[float]:
        """Return current parameters as a flat list."""
        return self._init_params()


__all__ = ["HybridQRNN"]
