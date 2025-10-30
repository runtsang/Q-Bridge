"""
Quantum‑variational estimator with a two‑qubit entangled ansatz and quantum kernel.
Provides training, prediction, and evaluation interfaces mirroring the classical counterpart.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeLocalBristol
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import Estimator
from qiskit_machine_learning.algorithms import NeuralNetwork
from qiskit.utils import algorithm_globals
from qiskit.providers import BackendV2
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliExpectation, StateFn, AerPauliExpectation
from qiskit.quantum_info import SparsePauliOp
import qiskit
import torch
import torch.nn as nn
import torch.optim as optim
import warnings


class EstimatorQNN:
    """
    Variational quantum circuit with entanglement and a quantum kernel.
    Uses a 2‑qubit RealAmplitudes ansatz with two layers and a classical optimizer.
    """

    def __init__(self, *, depth: int = 2, backend: Optional[BackendV2] = None, seed: int = 42) -> None:
        self.depth = depth
        self.seed = seed
        algorithm_globals.random_seed = self.seed
        self.backend = backend or FakeLocalBristol()
        self.circuit = self._build_circuit()
        self.observables = SparsePauliOp.from_list([("Z" * self.circuit.num_qubits, 1)])
        self.estimator = Estimator(
            backend=self.backend,
            estimator=QiskitEstimatorQNN(
                circuit=self.circuit,
                observables=self.observables,
                input_params=[self.circuit.parameters[0]],
                weight_params=self.circuit.parameters[1:],
                estimator=Estimator(),
            ),
        )
        self._optimizer = None
        self._weights = None

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        """
        Builds a 2‑qubit RealAmplitudes circuit with entanglement.
        """
        params = [Parameter(f"inp_{i}") for i in range(2)]
        circuit = RealAmplitudes(num_qubits=2, reps=self.depth, entanglement="full")
        circuit = circuit.bind_parameters({p: 0.0 for p in circuit.parameters})
        # Replace first two parameters with input parameters
        for i, p in enumerate(circuit.parameters[:2]):
            circuit.params[i] = params[i]
        return circuit

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """
        Train the variational circuit using a classical optimizer (Adam).
        """
        if X.shape[1]!= 2:
            raise ValueError("Input dimension must be 2 for this estimator.")
        self._optimizer = optim.Adam(self.estimator.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, target in zip(X, y):
                inputs = torch.tensor(x, dtype=torch.float32)
                target_t = torch.tensor([target], dtype=torch.float32)
                self._optimizer.zero_grad()
                pred = self.estimator(inputs)
                loss = criterion(pred, target_t)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(X)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – MSE: {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # Prediction & evaluation
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predictions for the provided inputs.
        """
        preds = []
        for x in X:
            pred = self.estimator(torch.tensor(x, dtype=torch.float32))
            preds.append(pred.item())
        return np.array(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Return MAE and RMSE on the provided data.
        """
        preds = self.predict(X)
        mae = np.mean(np.abs(preds - y))
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        return mae, rmse

    # ------------------------------------------------------------------
    # Quantum kernel helper
    # ------------------------------------------------------------------
    def quantum_kernel(self) -> QuantumKernel:
        """
        Return a QuantumKernel instance based on the same circuit and backend.
        """
        return QuantumKernel(
            circuit=self.circuit,
            feature_map=self.circuit,
            backend=self.backend,
            input_params=[self.circuit.parameters[0]],
            weight_params=self.circuit.parameters[1:],
        )

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.estimator.state_dict(), path)

    def load(self, path: str) -> None:
        self.estimator.load_state_dict(torch.load(path))

__all__ = ["EstimatorQNN"]
