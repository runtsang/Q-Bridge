"""Quantum QCNN hybrid model with convolution and pooling layers, parameter clipping, and a feature map."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QCNNGen520:
    """Quantum QCNN with parameter clipping inspired by photonic fraud‑detection layers."""
    def __init__(self, seed: int = 12345) -> None:
        self.seed = seed
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        np.random.seed(self.seed)
        estimator = Estimator()

        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(_clip(params[0], 5), 0)
            qc.ry(_clip(params[1], 5), 1)
            qc.cx(0, 1)
            qc.ry(_clip(params[2], 5), 1)
            qc.cx(1, 0)
            qc.rz(np.pi / 2, 0)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            params = ParameterVector(prefix, length=num_qubits * 3)
            for i in range(0, len(qubits), 2):
                sub = conv_circuit(params[3 * i : 3 * (i + 1)])
                qc.append(sub.to_instruction(), [qubits[i], qubits[i + 1]])
                qc.barrier()
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(_clip(params[0], 5), 0)
            qc.ry(_clip(params[1], 5), 1)
            qc.cx(0, 1)
            qc.ry(_clip(params[2], 5), 1)
            return qc

        def pool_layer(src, sink, prefix):
            num = len(src) + len(sink)
            qc = QuantumCircuit(num)
            params = ParameterVector(prefix, length=len(src) * 3)
            for i in range(len(src)):
                sub = pool_circuit(params[3 * i : 3 * (i + 1)])
                qc.append(sub.to_instruction(), [src[i], sink[i]])
                qc.barrier()
            return qc

        # Build ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        feature_map = ZFeatureMap(8)
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        qnn = EstimatorQNN(
            circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        return qnn

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Run the quantum neural network on classical data."""
        return self.qnn.predict(data)

__all__ = ["QCNNGen520"]
