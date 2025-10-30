"""Quantum QCNN implementation with deterministic and noisy evaluation."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List, Union

# Lightweight quantum estimator from the second reference
from.FastBaseEstimator import FastBaseEstimator

# ----------------------------------------------------------------------
# Helper functions for building the QCNN variational ansatz
# ----------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index + 3]), [source, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

# ----------------------------------------------------------------------
# Hybrid QCNN quantum model
# ----------------------------------------------------------------------
class QCNNHybridQNN:
    """Quantum QCNN that can evaluate expectation values deterministically or with shot noise."""
    def __init__(self, shots: int | None = None, seed: int | None = None) -> None:
        self.shots = shots
        self.seed = seed
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, range(8), inplace=True)
        self.circuit.compose(self.ansatz, range(8), inplace=True)

        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # EstimatorQNN for efficient, parameter‑vectorised evaluation
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator(),
        )

        # Deterministic fast estimator used for shot‑noise simulation
        self.fast_estimator = FastBaseEstimator(self.circuit)

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8, name="Ansatz")
        # First Convolution + Pooling
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        # Second Convolution + Pooling
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        # Third Convolution + Pooling
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
        return ansatz

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        If ``shots`` is provided, Gaussian shot noise with variance ``1/shots`` is added.
        """
        shots = shots if shots is not None else self.shots
        seed = seed if seed is not None else self.seed

        if shots is None:
            # Fast parameter‑vectorised evaluation via EstimatorQNN
            return self.qnn.predict(parameter_sets)

        # Deterministic evaluation
        raw = self.fast_estimator.evaluate(observables, parameter_sets)

        # Inject shot noise
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy

def QCNN() -> QCNNHybridQNN:
    """Factory returning a configurable hybrid QCNN quantum model."""
    return QCNNHybridQNN()

__all__ = ["QCNNHybridQNN", "QCNN"]
