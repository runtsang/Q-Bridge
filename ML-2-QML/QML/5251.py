from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Import anchor circuit factory
from.QuantumClassifierModel import build_classifier_circuit as build_qc_circuit
# Estimator utilities
from.FastBaseEstimator import FastBaseEstimator
# Quantum kernel
from.QuantumKernelMethod import KernalAnsatz, Kernel
# Quantum neural network example
from.EstimatorQNN import EstimatorQNN

class HybridClassifier:
    """
    Quantum‑classical hybrid classifier mirroring the classical interface.

    * Builds a parameterised circuit via :func:`build_qc_circuit`.
    * Uses :class:`FastBaseEstimator` for expectation‑value evaluation.
    * Optionally attaches a fixed quantum kernel and a QNN regressor.
    """

    def __init__(self, num_qubits: int, depth: int, gamma: float = 1.0,
                 shots: int | None = None, seed: int | None = None) -> None:
        self.circuit, self.encoding, self.weights, self.observables = \
            build_qc_circuit(num_qubits, depth)
        self.estimator = FastBaseEstimator(self.circuit)
        self.kernel = Kernel()
        self.qnn = EstimatorQNN()
        self.shots = shots
        self.seed = seed

    def predict(self, params: Sequence[float]) -> List[complex]:
        """Return expectation values for each observable."""
        return self.estimator.evaluate(self.observables, [params])[0]

    def evaluate(self, param_sets: Sequence[Sequence[float]],
                 observables: Iterable[BaseOperator] | None = None) -> List[List[complex]]:
        """Fast evaluation with optional shot noise (not implemented here)."""
        return self.estimator.evaluate(
            observables or self.observables,
            param_sets
        )

    def kernel_matrix(self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
        """Compute Gram matrix via the fixed quantum kernel."""
        return np.array([[self.kernel(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a])

    def qnn_predict(self, x: Sequence[float]) -> torch.Tensor:
        """Delegate to the quantum neural network regressor."""
        return self.qnn(torch.tensor(x, dtype=torch.float32))
