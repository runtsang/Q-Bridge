import numpy as np
import qiskit
from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCSelfAttentionModel:
    """
    Quantum QCNN with an embedded attention‑style circuit.
    The ansatz contains convolution, pooling, and a custom
    attention block that applies RX/RY/RZ rotations followed
    by a CRX entangling gate.
    """
    def __init__(self, n_qubits: int = 8, seed: int = 123):
        self.n_qubits = n_qubits
        self.estimator = Estimator()
        self.circuit = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Circuit primitives
    # ------------------------------------------------------------------
    def _conv_block(self, params):
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

    def _pool_block(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _attention_block(self, params):
        qc = QuantumCircuit(1)
        qc.rx(params[0], 0)
        qc.ry(params[1], 0)
        qc.rz(params[2], 0)
        return qc

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_block(params[i * 3 : (i + 2) * 3])
            qc.append(sub, [i, i + 1])
            qc.barrier()
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, sink in zip(sources, sinks):
            sub = self._pool_block(params[:3])
            params = params[3:]
            qc.append(sub, [src, sink])
            qc.barrier()
        return qc

    def _attention_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(num_qubits):
            sub = self._attention_block(params[i * 3 : (i + 1) * 3])
            qc.append(sub, [i])
        # Entangle neighboring qubits with a fixed CRX
        for i in range(num_qubits - 1):
            qc.crx(np.pi / 4, i, i + 1)
        return qc

    # ------------------------------------------------------------------
    # Build full ansatz
    # ------------------------------------------------------------------
    def _build_circuit(self):
        self.feature_map = ZFeatureMap(self.n_qubits)
        self.ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")

        # First convolutional layer
        self.ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)

        # First pooling layer
        self.ansatz.compose(
            self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True
        )

        # Second convolutional layer
        self.ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

        # Second pooling layer
        self.ansatz.compose(
            self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True
        )

        # Attention layer inserted after second pooling
        self.ansatz.compose(
            self._attention_layer(4, "a1"), list(range(4, 8)), inplace=True
        )

        # Third convolutional layer
        self.ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

        # Third pooling layer
        self.ansatz.compose(
            self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True
        )

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, range(self.n_qubits), inplace=True)
        circuit.compose(self.ansatz, range(self.n_qubits), inplace=True)
        return circuit

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the quantum neural network on the provided inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Input array of shape (batch, 8).

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return self.qnn.predict(inputs)

def QCNNSelfAttention() -> QCSelfAttentionModel:
    """
    Factory returning a configured QCSelfAttentionModel.
    """
    return QCSelfAttentionModel()
