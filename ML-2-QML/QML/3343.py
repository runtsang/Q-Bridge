"""Hybrid EstimatorQNN implemented with Qiskit and Pennylane.

This quantum module mirrors the classical EstimatorQNN but exposes a pure quantum
implementation that can be used as a drop‑in replacement or for educational
purposes. It demonstrates:
  • Parameterized circuits with Ry and CNOT entanglement.
  • Expectation value measurement of Pauli‑Z.
  • A simple classical post‑processing head that maps the expectation to a
    probability via a sigmoid (classification) or returns it directly
    (regression).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Qiskit backend
try:
    from qiskit import QuantumCircuit as QC, Aer, transpile, assemble
    from qiskit.providers.aer import AerSimulator
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False


class QuantumCircuitWrapper:
    """Variational circuit executed on a Pennylane device.

    The circuit contains a single Ry rotation on each qubit followed by a
    CNOT chain. The expectation of Pauli‑Z on the first qubit is returned.
    """
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self._node = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, theta: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RY(theta, wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def run(self, theta: torch.Tensor) -> torch.Tensor:
        return self._node(theta)


class QuantumCircuitQiskit:
    """Fallback implementation using Qiskit Aer simulator."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        if not HAS_QISKIT:
            raise RuntimeError("Qiskit is required for QuantumCircuitQiskit.")
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QC(n_qubits)
        self.theta = QC.Parameter("theta")

        for i in range(n_qubits):
            self.circuit.ry(self.theta, i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, theta: np.ndarray) -> np.ndarray:
        bound = self.circuit.bind_parameters({self.theta: theta})
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return np.sum((1 - 2 * states) * probs)


class HybridModule(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend: str = "pennylane", shots: int = 1024):
        super().__init__()
        if backend.lower() == "pennylane":
            self.circuit = QuantumCircuitWrapper(n_qubits, shots)
        elif backend.lower() == "qiskit":
            self.circuit = QuantumCircuitQiskit(n_qubits, shots)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Accept batch of angles
        angles = inputs.detach().cpu().numpy().flatten()
        exp_vals = np.array([self.circuit.run(a) for a in angles])
        return torch.tensor(exp_vals, device=inputs.device)


class QuantumEstimatorQNN(nn.Module):
    """A pure‑quantum version of EstimatorQNN.

    The network consists of a simple classical preprocessing head followed by
    a quantum expectation layer. The output is either a regression value or a
    binary probability distribution.
    """
    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int] | None = None,
        n_qubits: int = 2,
        backend: str = "pennylane",
        shots: int = 1024,
        task: str = "regression",
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [32]
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.prep = nn.Sequential(*layers)
        self.hybrid = HybridModule(n_qubits, backend, shots)
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.prep(x)
        angle = features.mean(dim=1, keepdim=True)
        exp_val = self.hybrid(angle)
        if self.task == "classification":
            prob = torch.sigmoid(exp_val)
            return torch.cat((prob, 1 - prob), dim=-1)
        return exp_val.squeeze(-1)


__all__ = ["QuantumCircuitWrapper", "QuantumCircuitQiskit", "HybridModule", "QuantumEstimatorQNN"]
