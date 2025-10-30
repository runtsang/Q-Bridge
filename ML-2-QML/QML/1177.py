"""Quantum QCNN model with a hybrid variational circuit and classical post‑processing.

The `QCNNModel` class builds a full variational circuit that mimics the
convolution‑pooling structure of the classical QCNN.  It uses Qiskit
EstimatorQNN to evaluate expectation values and a small classical
neural network to map the quantum output to a probability.  The
architecture is fully parameterised and supports gradient‑based
optimisation via the `qiskit_machine_learning` optimisers.

Public factory function `QCNN()` returns an instance ready for training.
"""

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

__all__ = ["QCNN", "QCNNModel"]

class QCNNModel:
    """Hybrid QCNN: variational quantum circuit + classical head."""
    def __init__(self, n_qubits: int = 8, feature_dim: int = 8,
                 hidden_dim: int = 4, seed: int = 42):
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seed = seed

        # Feature map: encode classical data into quantum state
        self.feature_map = ZFeatureMap(feature_dim, reps=1, entanglement='full')
        self.feature_map.seed(seed)

        # Ansatz: convolution‑pooling style variational circuit
        self.ansatz = self._build_ansatz()

        # Observable: single‑qubit Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        # Estimator for expectation values
        self.estimator = Estimator()

        # Compose full circuit: feature map + ansatz
        circuit = QuantumCircuit(n_qubits)
        circuit.compose(self.feature_map, range(n_qubits), inplace=True)
        circuit.compose(self.ansatz, range(n_qubits), inplace=True)

        # Hybrid QNN
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator
        )

        # Classical post‑processing head (small neural network)
        self.classical_head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct a convolution‑pooling style ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        # Convolutional layers
        for layer in range(3):
            for i in range(0, self.n_qubits, 2):
                self._conv_block(qc, i, i + 1)
            # Pooling: entangle pairs (classical head handles pooling)
            if layer < 2:
                for i in range(0, self.n_qubits, 4):
                    self._pool_block(qc, i, i + 2)
        return qc

    def _conv_block(self, qc: QuantumCircuit, q1: int, q2: int):
        """Convolution block between two qubits."""
        theta = ParameterVector(f"θ_{q1}_{q2}", length=3)
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(theta[0], q1)
        qc.ry(theta[1], q2)
        qc.cx(q1, q2)
        qc.ry(theta[2], q2)
        qc.cx(q2, q1)
        qc.rz(np.pi / 2, q1)

    def _pool_block(self, qc: QuantumCircuit, q1: int, q2: int):
        """Pooling block that entangles two qubits."""
        phi = ParameterVector(f"φ_{q1}_{q2}", length=3)
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(phi[0], q1)
        qc.ry(phi[1], q2)
        qc.cx(q1, q2)
        qc.ry(phi[2], q2)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Compute the model output for a batch of classical data."""
        # Expectation values from the quantum circuit
        exp_vals = self.qnn.predict(x)
        exp_tensor = torch.tensor(exp_vals, dtype=torch.float32).unsqueeze(-1)
        # Classical head
        return self.classical_head(exp_tensor)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return raw probability predictions."""
        return self.forward(x).detach().numpy()

def QCNN() -> QCNNModel:
    """Factory returning a default QCNNModel instance."""
    return QCNNModel()
