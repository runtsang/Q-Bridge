"""Variational quantum regressor with training loop.

Features:
- Two‑qubit circuit with parameter‑shift encoding and entanglement.
- Multiple observables (Pauli‑Z on each qubit) as outputs.
- Gradient‑based training using Adam and MSE loss.
- Compatibility with the original EstimatorQNN factory pattern.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import PauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
import torch
from torch import nn
from torch.optim import Adam

class EstimatorQNNModel:
    """Variational quantum neural network for regression."""
    def __init__(self, num_qubits: int = 2, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self._build()

    def _build(self) -> None:
        # Parameters
        self.input_param = Parameter("x")
        self.weight_params = [Parameter(f"w{i}") for i in range(self.num_qubits * self.depth)]

        # Circuit
        qc = QuantumCircuit(self.num_qubits)
        # Simple Ry encoding of the scalar input on qubit 0
        qc.ry(self.input_param, 0)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(self.weight_params[idx], q)
                idx += 1
                qc.rz(self.weight_params[idx], q)
                idx += 1
            # Entanglement
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
        self.circuit = qc

        # Observables – Pauli‑Z on each qubit
        self.observables = [PauliOp.from_label("Z") for _ in range(self.num_qubits)]

        # Estimator
        self.estimator = StatevectorEstimator()
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=[self.input_param],
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, x: float | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return expectation values for a single input."""
        if isinstance(x, (list, np.ndarray)):
            x = np.asarray(x).astype(np.float32)
            return torch.stack([self.qnn.forward([float(val)]) for val in x])
        return self.qnn.forward([float(x)])

    def train(
        self,
        X: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Gradient‑based training of the variational parameters."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        opt = Adam(self.qnn.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            opt.zero_grad()
            preds = torch.stack([self.qnn.forward([float(xi)]) for xi in X])
            loss = loss_fn(preds, torch.tensor(y))
            loss.backward()
            opt.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} loss={loss.item():.4f}")

    def evaluate(self, X: list[float] | np.ndarray) -> np.ndarray:
        """Return predictions as a NumPy array."""
        X = np.asarray(X, dtype=np.float32)
        return torch.stack([self.qnn.forward([float(xi)]) for xi in X]).detach().numpy()

def EstimatorQNN() -> EstimatorQNNModel:
    """Factory function mirroring the original API."""
    return EstimatorQNNModel()

__all__ = ["EstimatorQNN"]
