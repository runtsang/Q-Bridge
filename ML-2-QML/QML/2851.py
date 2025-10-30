"""Quantum hybrid network mirroring the classical HybridFCQCNN architecture."""

import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class HybridFCQCNN:
    """
    A quantum neural network that mirrors the classical HybridFCQCNN architecture.
    It encodes input data via a Z‑feature map, applies a convolution‑pooling ansatz,
    and returns the expectation value of a single‑qubit Pauli‑Z observable.
    """

    def __init__(self, num_qubits: int = 8, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits=num_qubits)

        # Build ansatz
        self.ansatz = self._build_ansatz()

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator(),
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct a convolution‑pooling ansatz similar to the QCNN helper."""

        def conv_unit(params, q1, q2):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, q2)
            qc.cx(q2, q1)
            qc.rz(params[0], q1)
            qc.ry(params[1], q2)
            qc.cx(q1, q2)
            qc.ry(params[2], q2)
            qc.cx(q2, q1)
            qc.rz(np.pi / 2, q1)
            return qc

        def pool_unit(params, q1, q2):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, q2)
            qc.cx(q2, q1)
            qc.rz(params[0], q1)
            qc.ry(params[1], q2)
            qc.cx(q1, q2)
            qc.ry(params[2], q2)
            return qc

        qc = QuantumCircuit(self.num_qubits)

        # First convolutional layer: pairs (0,1), (2,3), (4,5), (6,7)
        for i in range(0, self.num_qubits, 2):
            params = ParameterVector(f"c1_{i}", length=3)
            qc.append(conv_unit(params, i, i + 1), [i, i + 1])
            qc.barrier()

        # First pooling layer: pairs (0,2), (1,3), (4,6), (5,7)
        for i in range(0, self.num_qubits, 4):
            params = ParameterVector(f"p1_{i}", length=3)
            qc.append(pool_unit(params, i, i + 2), [i, i + 2])
            qc.barrier()

        # Second convolutional layer: pairs (0,1), (2,3)
        for i in range(0, self.num_qubits // 2, 2):
            params = ParameterVector(f"c2_{i}", length=3)
            qc.append(conv_unit(params, i, i + 1), [i, i + 1])
            qc.barrier()

        # Second pooling layer: pairs (0,2), (1,3)
        for i in range(0, self.num_qubits // 2, 4):
            params = ParameterVector(f"p2_{i}", length=3)
            qc.append(pool_unit(params, i, i + 2), [i, i + 2])
            qc.barrier()

        # Third convolutional layer: pair (0,1)
        params = ParameterVector("c3_0", length=3)
        qc.append(conv_unit(params, 0, 1), [0, 1])
        qc.barrier()

        # Third pooling layer: pair (0,1)
        params = ParameterVector("p3_0", length=3)
        qc.append(pool_unit(params, 0, 1), [0, 1])
        qc.barrier()

        return qc

    def run(self, input_vector: np.ndarray) -> np.ndarray:
        """Predict the expectation value for a single input vector."""
        return self.qnn.predict(inputs=[input_vector])[0]


def HybridFCQCNNFactory() -> HybridFCQCNN:
    """Convenience factory returning a default configured instance."""
    return HybridFCQCNN()
