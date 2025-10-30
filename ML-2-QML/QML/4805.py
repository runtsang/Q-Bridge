from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit import assemble, transpile
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

class ConvGen259:
    """Quantum‑inspired convolutional network.

    The class builds a QCNN‑style ansatz composed of
    parameterised convolution and pooling layers followed
    by a feature map.  It uses EstimatorQNN to evaluate
    expectation values and can be called with a 2‑D image
    patch to produce a binary probability.
    """
    def __init__(self, shots: int = 100) -> None:
        self.shots = shots
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        # Build QNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, name: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(name, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], name: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(name, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            qc.append(self._pool_circuit(params[param_index:param_index+3]), [src, sink])
            qc.barrier()
            param_index += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8)
        # First convolution & pooling
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3],[4,5,6,7], "p1"), list(range(8)), inplace=True)
        # Second convolution & pooling
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4,8)), inplace=True)
        ansatz.compose(self._pool_layer([0,1],[2,3], "p2"), list(range(4,8)), inplace=True)
        # Third convolution & pooling
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6,8)), inplace=True)
        ansatz.compose(self._pool_layer([0],[1], "p3"), list(range(6,8)), inplace=True)
        return ansatz

    def run(self, data: np.ndarray) -> float:
        """Evaluate the QCNN on a single 2‑D patch.

        Args:
            data: 1‑D array of length 8 (flattened 2‑D patch).

        Returns:
            float: probability of class 1.
        """
        if data.ndim > 1:
            data = data.flatten()
        # Expectation values for a single sample
        preds = self.qnn.predict(np.array([data]))
        return float(preds[0][0])

__all__ = ["ConvGen259"]
