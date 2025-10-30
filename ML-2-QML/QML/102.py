"""Quantum convolutional neural network with parameter‑shift gradient and noise awareness."""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import Adam
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Tuple


class QCNNModel(BaseEstimator, ClassifierMixin):
    """A hybrid QCNN model that wraps an EstimatorQNN and provides sklearn‑style API."""

    def __init__(
        self,
        n_qubits: int = 8,
        feature_map_depth: int = 1,
        ansatz_depth: int = 3,
        learning_rate: float = 0.01,
        epochs: int = 50,
        batch_size: int = 16,
        seed: int = 42,
        noise_model: NoiseModel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n_qubits: int
            Number of qubits in the quantum circuit.
        feature_map_depth: int
            Depth of the ZFeatureMap (controls entanglement).
        ansatz_depth: int
            Number of convolution–pooling cycles in the ansatz.
        learning_rate: float
            Learning rate for the Adam optimiser.
        epochs: int
            Number of training epochs.
        batch_size: int
            Batch size for stochastic optimisation.
        seed: int
            Random seed for reproducibility.
        noise_model: NoiseModel | None
            Optional noise model to run the circuit on a noisy simulator.
        """
        self.n_qubits = n_qubits
        self.feature_map_depth = feature_map_depth
        self.ansatz_depth = ansatz_depth
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.noise_model = noise_model

        # Build the quantum circuit and EstimatorQNN
        self.feature_map = ZFeatureMap(n_qubits, reps=feature_map_depth, entanglement="linear")
        self.circuit = self._build_ansatz()
        self.obs = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])
        self.estimator = Estimator(
            backend=(
                AerSimulator(noise_model=self.noise_model)
                if self.noise_model
                else AerSimulator()
            )
        )
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.obs,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

        self.optimizer = Adam(self.qnn, lr=self.learning_rate)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution unit as in the original design."""
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling unit."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Constructs a QCNN‑style ansatz with alternating conv/pool layers."""
        qc = QuantumCircuit(self.n_qubits)
        qubits = list(range(self.n_qubits))

        # Convolution–pooling cycles
        for cycle in range(self.ansatz_depth):
            # Convolution layer
            conv_params = ParameterVector(f"c{cycle}_", length=(self.n_qubits // 2) * 3)
            conv_inst = self._conv_circuit(conv_params).to_instruction()
            for i in range(0, self.n_qubits, 2):
                qc.append(conv_inst, [i, i + 1])
                qc.barrier()

            # Pooling layer
            pool_params = ParameterVector(f"p{cycle}_", length=(self.n_qubits // 2) * 3)
            pool_inst = self._pool_circuit(pool_params).to_instruction()
            for i in range(0, self.n_qubits, 2):
                qc.append(pool_inst, [i, i + 1])
                qc.barrier()

        return qc

    def _prepare_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to torch tensors and split into batches."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return X_t, y_t

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QCNNModel":
        """Fit the hybrid QCNN using Adam optimisation."""
        X_t, y_t = self._prepare_batch(X, y)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.qnn(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if epoch % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0/1)."""
        self.qnn.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        probs = self.qnn(X_t).detach().numpy().reshape(-1)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates."""
        X_t = torch.tensor(X, dtype=torch.float32)
        probs = self.qnn(X_t).detach().numpy().reshape(-1)
        return np.vstack([1 - probs, probs]).T

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        preds = self.predict(X)
        return (preds == y).mean()


__all__ = ["QCNNModel"]
