"""Quantum QCNN implementation with fast estimation and shot‑noise emulation."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import ZFeatureMap
from typing import Iterable, List, Sequence, Tuple

class QCNNUnified:
    """Quantum QCNN wrapper exposing a deterministic and noisy evaluation API."""
    def __init__(self, seed: int = 12345) -> None:
        self.seed = seed
        algorithm_globals.random_seed = seed
        self.estimator = QiskitEstimator()
        self.qnn = self._build_qnn()

    # ----- internal helpers -----
    def _conv_circuit(self, params: Sequence[float]) -> QuantumCircuit:
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

    def _pool_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, name: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(f"{name}_θ", length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(param_vec[idx : idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, num_qubits: int, name: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(f"{name}_θ", length=num_qubits * 3)
        idx = 0
        for q in range(num_qubits - 1):
            sub = self._pool_circuit(param_vec[idx : idx + 3])
            qc.append(sub, [q, q + 1])
            qc.barrier()
            idx += 3
        return qc

    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")

        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer(8, "p1"), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer(4, "p2"), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer(2, "p3"), inplace=True)

        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    # ----- public API -----
    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Batch‑evaluate the QCNN; if shots is given, add Gaussian noise per shot."""
        batch = [dict(zip(self.qnn.input_params, inp)) for inp in inputs]
        results = self.qnn.predict(batch)
        if shots is None:
            return results
        rng = np.random.default_rng(self.seed)
        noisy = []
        for row in results:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

# ----- quantum classifier builder -----
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Return a shallow variational ansatz, encoding and observables."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ----- factory -----
def QCNN() -> tuple[QCNNUnified]:
    """Convenience constructor returning a quantum QCNN wrapper."""
    return QCNNUnified(),

__all__ = ["QCNNUnified", "QCNN", "build_classifier_circuit"]
