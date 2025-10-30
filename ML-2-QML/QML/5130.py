import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn

class QuantumLinearHead(nn.Module):
    """
    Tiny classical head that takes a single quantum expectation value
    and turns it into a scalar prediction.
    """
    def __init__(self, weight: float = 0.5, bias: float = 0.1):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, expectation: np.ndarray) -> float:
        return float(self.weight * expectation[0] + self.bias)


class HybridEstimatorQNN:
    """
    Quantum neural network that combines a QCNN ansatz with a classical linear head.

    The circuit consists of:
    1. An 8‑qubit Z‑feature map.
    2. Three layers of 2‑qubit convolution + pooling blocks (QCNN style).
    3. A single‑qubit Pauli‑Z observable on the most‑significant qubit.

    The expectation value of this observable is passed through a tiny
    linear layer to produce the final prediction.
    """

    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.estimator = StatevectorEstimator()
        self.feature_map = self._build_feature_map()
        self.circuit = self._build_ansatz()
        self.observables = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator
        )
        self.head = QuantumLinearHead()

    def _build_feature_map(self) -> QuantumCircuit:
        """Simple 8‑qubit Z‑feature map."""
        fmap = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            fmap.rz(2 * np.pi * i / self.num_qubits, i)
        return fmap

    def _conv_block(self, qubits: list[int]) -> QuantumCircuit:
        """One 2‑qubit convolution block used in the QCNN ansatz."""
        params = ParameterVector("θ", length=len(qubits) // 2 * 3)
        qc = QuantumCircuit(len(qubits))
        idx = 0
        for i in range(0, len(qubits), 2):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(params[idx], 0)
            sub.ry(params[idx + 1], 1)
            sub.cx(0, 1)
            sub.ry(params[idx + 2], 1)
            sub.cx(1, 0)
            sub.rz(np.pi / 2, 0)
            qc.append(sub, [qubits[i], qubits[i + 1]])
            idx += 3
        return qc

    def _pool_block(self, qubits: list[int]) -> QuantumCircuit:
        """One 2‑qubit pooling block."""
        params = ParameterVector("θ", length=len(qubits) // 2 * 3)
        qc = QuantumCircuit(len(qubits))
        idx = 0
        for i in range(0, len(qubits), 2):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(params[idx], 0)
            sub.ry(params[idx + 1], 1)
            sub.cx(0, 1)
            sub.ry(params[idx + 2], 1)
            qc.append(sub, [qubits[i], qubits[i + 1]])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full QCNN ansatz."""
        fmap = self._build_feature_map()
        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.append(fmap, range(self.num_qubits))

        # First convolution + pooling layer
        ansatz.append(self._conv_block(list(range(8))), range(8))
        ansatz.append(self._pool_block([0, 1, 2, 3, 4, 5, 6, 7]), range(8))

        # Second convolution + pooling layer
        ansatz.append(self._conv_block([0, 1, 2, 3]), range(4))
        ansatz.append(self._pool_block([0, 1, 2, 3]), range(4))

        # Third convolution + pooling layer
        ansatz.append(self._conv_block([0, 1]), range(2))
        ansatz.append(self._pool_block([0, 1]), range(2))

        return ansatz.decompose()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the hybrid QNN on a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray of shape (batch, num_features)
            Classical features that will be encoded by the feature map.

        Returns
        -------
        np.ndarray of shape (batch,)
            Hybrid predictions.
        """
        batch_size = inputs.shape[0]
        predictions = []
        for i in range(batch_size):
            # Bind the feature‑map parameters to the i‑th sample
            binds = {p: val for p, val in zip(self.feature_map.parameters, inputs[i])}
            # Run the EstimatorQNN to get the expectation value
            result = self.qnn.predict(parameter_binds=[binds])
            expectation = result[0].expectation
            # Classical head
            pred = self.head(expectation)
            predictions.append(pred)
        return np.array(predictions).squeeze()

__all__ = ["HybridEstimatorQNN"]
