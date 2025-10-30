"""Quantum QCNN with a deeper ZZ‑feature map and multi‑layer ansatz."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance


class QCNNModel:
    """Quantum QCNN wrapped as a PyTorch‑style model."""

    def __init__(
        self,
        num_qubits: int = 8,
        feature_depth: int = 2,
        ansatz_depth: int = 4,
        seed: int = 12345,
    ) -> None:
        self.num_qubits = num_qubits
        self.feature_map = ZZFeatureMap(num_qubits, reps=feature_depth)
        self.ansatz = self._build_ansatz(ansatz_depth)
        self.observables = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.estimator = Estimator(
            backend=AerSimulator(method="statevector", seed_simulator=seed)
        )
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self, depth: int) -> QuantumCircuit:
        """Construct a layered ansatz with alternating RY and CZ gates."""
        qc = QuantumCircuit(self.num_qubits)
        for d in range(depth):
            # Parameterised rotation on each qubit
            for q in range(self.num_qubits):
                qc.ry(ParameterVector(f"ry_{d}_{q}", 1)[0], q)
            # Entangling layer
            for q in range(0, self.num_qubits - 1, 2):
                qc.cz(q, q + 1)
            for q in range(1, self.num_qubits - 1, 2):
                qc.cz(q, q + 1)
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the model output for a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch_size, num_qubits) with values in [-1, 1].

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (batch_size,).
        """
        # Ensure inputs are in the expected range
        inputs = np.clip(inputs, -1.0, 1.0)
        # Evaluate the QNN
        return self.qnn.predict(inputs).reshape(-1)

    def circuit(self) -> QuantumCircuit:
        """Return the full circuit (feature map + ansatz)."""
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        full_circuit.compose(self.ansatz, range(self.num_qubits), inplace=True)
        return full_circuit


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
