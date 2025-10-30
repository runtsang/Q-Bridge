"""Quantum latent encoder for the hybrid autoencoder using Qiskit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridAutoencoder:
    """Quantum circuit that maps classical features to a latent vector
    using a convolution‑style ansatz inspired by QCNN."""
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Feature map to encode input data into a quantum state
        self.feature_map = ZFeatureMap(num_qubits=input_dim, reps=1)
        self.input_params = self.feature_map.parameters

        # Build the convolution‑pooling ansatz
        self.ansatz = self._build_ansatz(latent_dim)
        self.weight_params = self.ansatz.parameters

        # Observables: Z on each latent qubit
        self.observables = [
            SparsePauliOp.from_list([("Z" + "I" * (i) + "I" * (latent_dim - i - 1), 1)])
            for i in range(latent_dim)
        ]

        # Construct the full circuit
        self.circuit = qiskit.QuantumCircuit(input_dim)
        self.circuit.compose(self.feature_map, range(input_dim), inplace=True)
        # The ansatz acts on the first latent_dim qubits
        self.circuit.compose(self.ansatz, range(latent_dim), inplace=True)

        # Estimator and QNN
        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def _build_ansatz(self, num_qubits: int) -> qiskit.QuantumCircuit:
        """Construct a QCNN‑style ansatz with two convolution‑pooling layers."""
        conv = self._conv_layer(num_qubits, prefix="c1")
        pool = self._pool_layer(num_qubits, prefix="p1")
        ansatz = qiskit.QuantumCircuit(num_qubits)
        ansatz.compose(conv, range(num_qubits), inplace=True)
        ansatz.compose(pool, range(num_qubits), inplace=True)
        return ansatz

    def _conv_layer(self, num_qubits: int, prefix: str) -> qiskit.QuantumCircuit:
        """Apply a QCNN convolution block to each adjacent pair of qubits."""
        qc = qiskit.QuantumCircuit(num_qubits)
        param_vec = qiskit.circuit.ParameterVector(prefix, length=num_qubits * 3 // 2)
        idx = 0
        for i in range(0, num_qubits - 1, 2):
            sub = self._conv_circuit(param_vec[idx : idx + 3])
            qc.compose(sub, [i, i + 1], inplace=True)
            idx += 3
        return qc

    def _pool_layer(self, num_qubits: int, prefix: str) -> qiskit.QuantumCircuit:
        """Apply a QCNN pooling block to each adjacent pair of qubits."""
        qc = qiskit.QuantumCircuit(num_qubits)
        param_vec = qiskit.circuit.ParameterVector(prefix, length=num_qubits * 3 // 2)
        idx = 0
        for i in range(0, num_qubits - 1, 2):
            sub = self._pool_circuit(param_vec[idx : idx + 3])
            qc.compose(sub, [i, i + 1], inplace=True)
            idx += 3
        return qc

    def _conv_circuit(self, params: qiskit.circuit.ParameterVector) -> qiskit.QuantumCircuit:
        """Single QCNN convolution circuit."""
        qc = qiskit.QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _pool_circuit(self, params: qiskit.circuit.ParameterVector) -> qiskit.QuantumCircuit:
        """Single QCNN pooling circuit."""
        qc = qiskit.QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute latent vector for a batch of feature vectors."""
        # Expect x shape (batch, input_dim)
        return self.qnn.forward(x)

__all__ = ["HybridAutoencoder"]
