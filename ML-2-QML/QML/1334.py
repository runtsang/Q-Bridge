"""Quantum convolutional neural network implemented with Qiskit and EstimatorQNN."""

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer.noise import NoiseModel, depolarizing_error

class QCNNModel(nn.Module):
    """
    Quantum convolutional neural network implemented with Qiskit and EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit (must be even for pairing).
    depth : int
        Number of convolution‑pooling stages.
    noise_level : float | None, optional
        Depolarizing error probability added to each two‑qubit gate.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        depth: int = 3,
        noise_level: float | None = None,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.noise_level = noise_level

        # Feature map
        self.feature_params = ParameterVector("x", length=num_qubits)
        self.feature_map = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self.feature_map.rz(self.feature_params[i], i)
            self.feature_map.h(i)

        # Ansatz
        self.params = ParameterVector("θ", length=self._param_length())
        self.ansatz = QuantumCircuit(num_qubits)
        self._build_ansatz()

        # Full circuit
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

        # Observable
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator with optional noise
        if noise_level is not None:
            noise_model = NoiseModel()
            error = depolarizing_error(noise_level, 2)
            noise_model.add_all_qubit_quantum_error(error, ["cx"])
            estimator = Estimator(noise_model=noise_model)
        else:
            estimator = Estimator()

        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_params,
            weight_params=self.params,
            estimator=estimator,
        )

    def _param_length(self) -> int:
        """Total number of trainable parameters for the ansatz."""
        total = 0
        for stage in range(self.depth):
            pairs = self.num_qubits // (2 ** stage)
            total += pairs * 6  # 3 params per pair for conv + 3 for pool
        return total

    def _build_ansatz(self) -> None:
        idx = 0
        qubits = list(range(self.num_qubits))
        for stage in range(self.depth):
            pairs = len(qubits) // 2
            # Convolution block
            for i in range(pairs):
                self.ansatz.rz(self.params[idx], qubits[2 * i]); idx += 1
                self.ansatz.ry(self.params[idx], qubits[2 * i + 1]); idx += 1
                self.ansatz.cx(qubits[2 * i], qubits[2 * i + 1]); idx += 1
            # Pooling block
            for i in range(pairs):
                self.ansatz.rz(self.params[idx], qubits[2 * i]); idx += 1
                self.ansatz.ry(self.params[idx], qubits[2 * i + 1]); idx += 1
                self.ansatz.cx(qubits[2 * i], qubits[2 * i + 1]); idx += 1
            # Reduce qubits for next stage
            qubits = qubits[:pairs]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the QCNN output probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, num_qubits).

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch, 1).
        """
        probs = torch.sigmoid(self.qnn(inputs))
        return probs.unsqueeze(-1)

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Predict probabilities on CPU numpy arrays.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input data of shape (n_samples, num_qubits).

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).cpu().numpy().flatten()

def QCNN() -> QCNNModel:
    """Factory returning a default-configured :class:`QCNNModel`."""
    return QCNNModel(noise_level=0.01)

__all__ = ["QCNNModel", "QCNN"]
