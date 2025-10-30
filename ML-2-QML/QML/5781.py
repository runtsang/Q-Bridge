"""Hybrid quantum‑classical autoencoder with QCNN‑style quantum encoder and classical decoder."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class HybridQuantumAutoencoder:
    """Quantum encoder that produces a latent vector via a QCNN‑style ansatz,
    followed by a simple classical decoder."""

    def __init__(self, input_dim: int, latent_dim: int = 32, num_qubits: int | None = None):
        algorithm_globals.random_seed = 42
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits or input_dim  # one qubit per input feature

        # Feature map
        self.feature_map = ZFeatureMap(self.num_qubits)

        # QCNN ansatz
        self.ansatz = self._build_qcnn_ansatz(self.num_qubits)

        # Estimator for expectation values
        self.estimator = Estimator()

        # Observable to extract a single latent value per qubit
        observables = [SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])] * self.latent_dim

        # QNN that maps classical input to expectation value vector
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_qcnn_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Construct a QCNN‑style ansatz with convolution and pooling layers."""
        def conv_layer(qc: QuantumCircuit, params: list):
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi / 2, 0)
            return qc

        qc = QuantumCircuit(num_qubits)
        # Convolutional layers on adjacent qubit pairs
        for i in range(0, num_qubits, 2):
            params = [f"θ{i}_{j}" for j in range(3)]
            qc = qc.compose(conv_layer(qc, params), [i, i + 1])
        # Pooling layer (another conv on adjacent pairs)
        for i in range(0, num_qubits - 1, 2):
            params = [f"φ{i}_{j}" for j in range(3)]
            qc = qc.compose(conv_layer(qc, params), [i, i + 1])
        return qc

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Return latent vector for given classical data."""
        # EstimatorQNN expects 2‑D array of shape (batch, input_dim)
        preds = self.qnn.predict(inputs)
        # preds is a list of expectation values per sample; flatten to (batch, latent_dim)
        return np.array(preds, dtype=np.float32)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Simple linear decoder (replaceable with a PyTorch MLP)."""
        W = np.random.randn(self.latent_dim, self.input_dim).astype(np.float32)
        return latents @ W.T

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(inputs))
