"""Quantum hybrid QCNN + LSTM implementation.

The class builds a variational circuit that first encodes the input using a
ZFeatureMap, then applies a QCNN ansatz built from convolution and pooling
layers, followed by a simple quantum LSTM ansatz that updates hidden
information across time steps.  The whole circuit is wrapped in an
EstimatorQNN for training with a parameter‑shifting gradient.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import numpy as np

__all__ = ["QCNNQLSTMHybrid"]

class QCNNQLSTMHybrid:
    """
    Quantum hybrid of QCNN and LSTM.

    Parameters
    ----------
    input_dim : int
        Number of qubits for the feature map (must match n_qubits).
    hidden_dim : int
        Size of the hidden state encoded in the quantum LSTM.
    n_qubits : int
        Total number of qubits used in the circuit.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.estimator = Estimator()

        # Feature map for input encoding
        self.feature_map = ZFeatureMap(num_qubits=self.n_qubits, reps=1)

        # Build QCNN ansatz
        self.cnn_ansatz = self._build_cnn_ansatz()

        # Build quantum LSTM ansatz
        self.lstm_ansatz = self._build_lstm_ansatz()

        # Combine into full circuit
        self.circuit = self._build_full_circuit()

        # Observable: single Z on first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        # Create EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.cnn_ansatz.parameters + self.lstm_ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Convolution layer for two qubits."""
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

    def _build_pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Pooling layer for two qubits."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _build_conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Convolution layer composed of two‑qubit conv circuits."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(self._build_conv_circuit(params[i * 3 : (i + 2) * 3]), [i, i + 1], inplace=True)
        return qc

    def _build_pool_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Pooling layer composed of two‑qubit pool circuits."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(self._build_pool_circuit(params[i * 3 : (i + 2) * 3]), [i, i + 1], inplace=True)
        return qc

    def _build_cnn_ansatz(self) -> QuantumCircuit:
        """Builds the QCNN ansatz from conv and pool layers."""
        qc = QuantumCircuit(self.n_qubits)
        # First convolution layer
        qc.compose(self._build_conv_layer(self.n_qubits, "c1"), range(self.n_qubits), inplace=True)
        # First pooling layer
        qc.compose(self._build_pool_layer(self.n_qubits // 2, "p1"), range(self.n_qubits), inplace=True)
        # Second convolution layer
        qc.compose(self._build_conv_layer(self.n_qubits // 2, "c2"), range(self.n_qubits // 2), inplace=True)
        # Second pooling layer
        qc.compose(self._build_pool_layer(self.n_qubits // 4, "p2"), range(self.n_qubits // 2), inplace=True)
        return qc

    def _build_lstm_ansatz(self) -> QuantumCircuit:
        """Simple variational ansatz representing quantum LSTM gates."""
        qc = QuantumCircuit(self.n_qubits)
        # For each hidden qubit, apply a trainable rotation
        for i in range(self.hidden_dim):
            rc = ParameterVector(f"lstm_rx_{i}", 1)
            qc.rx(rc[0], i)
            rc = ParameterVector(f"lstm_ry_{i}", 1)
            qc.ry(rc[0], i)
        return qc

    def _build_full_circuit(self) -> QuantumCircuit:
        """Combine feature map, QCNN ansatz and quantum LSTM ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.feature_map, range(self.n_qubits), inplace=True)
        qc.compose(self.cnn_ansatz, range(self.n_qubits), inplace=True)
        qc.compose(self.lstm_ansatz, range(self.n_qubits), inplace=True)
        return qc
