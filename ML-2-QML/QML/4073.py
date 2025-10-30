"""Hybrid quantum-quantum binary classifier with fraud detection and self‑attention circuits.

This module implements a PyTorch model that delegates its final decision to
parameterised quantum circuits built with Qiskit.  The network mirrors the
classical architecture above while replacing the fraud‑detection and
self‑attention blocks with their quantum counterparts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter

# Quantum fraud‑detection layer -----------------------------------------

class QuantumFraudLayer:
    """Two‑qubit circuit emulating the fraud‑detection block."""
    def __init__(self, backend, shots: int = 100):
        self.backend = backend
        self.shots = shots
        self.theta1 = Parameter("theta1")
        self.theta2 = Parameter("theta2")

        self._circuit = qiskit.QuantumCircuit(2)
        self._circuit.h(0)
        self._circuit.h(1)
        self._circuit.ry(self.theta1, 0)
        self._circuit.ry(self.theta2, 1)
        self._circuit.cx(0, 1)
        self._circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        """Return expectation value <Z> on qubit 0."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta1: params[0], self.theta2: params[1]}],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for outcome, cnt in counts.items():
            z = 1 if outcome[0] == "0" else -1
            exp += z * cnt
        return exp / self.shots

# Quantum self‑attention block -----------------------------------------

class QuantumSelfAttention:
    """Self‑attention style block implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.rotation_params = [Parameter(f"theta_{i}") for i in range(n_qubits * 3)]
        self.entangle_params = [Parameter(f"phi_{i}") for i in range(n_qubits - 1)]

        self._circuit = qiskit.QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self._circuit.rx(self.rotation_params[3 * i], i)
            self._circuit.ry(self.rotation_params[3 * i + 1], i)
            self._circuit.rz(self.rotation_params[3 * i + 2], i)
        for i in range(n_qubits - 1):
            self._circuit.crx(self.entangle_params[i], i, i + 1)
        self._circuit.measure_all()

    def run(self, rotation_vals: np.ndarray, entangle_vals: np.ndarray) -> np.ndarray:
        """Return Z‑expectation values for each qubit."""
        bind = {p: v for p, v in zip(self.rotation_params, rotation_vals)}
        bind.update({p: v for p, v in zip(self.entangle_params, entangle_vals)})
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[bind],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = np.zeros(self.n_qubits)
        for outcome, cnt in counts.items():
            for i in range(self.n_qubits):
                z = 1 if outcome[self.n_qubits - 1 - i] == "0" else -1
                exp[i] += z * cnt
        return exp / self.shots

# Main classifier --------------------------------------------------------

class HybridQuantumBinaryClassifier(nn.Module):
    """
    Quantum‑enhanced classifier that mirrors the classical architecture.
    The final decision is derived from a quantum fraud‑detection circuit
    followed by a quantum self‑attention block and a differentiable sigmoid.
    """

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)

        # Classical projection to 2‑dim for the quantum fraud layer
        self.fraud_proj = nn.Linear(84, 2)

        # Quantum fraud detection circuit
        backend = Aer.get_backend("aer_simulator")
        self.quantum_fraud = QuantumFraudLayer(backend, shots=100)

        # Quantum self‑attention
        self.quantum_attention = QuantumSelfAttention(n_qubits=4, shots=1024)

        # Final head
        self.fc3 = nn.Linear(84 + 1 + 4, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))

        # Quantum fraud detection
        proj = self.fraud_proj(x).detach().cpu().numpy()
        fraud_exp = self.quantum_fraud.run(proj[0])
        fraud_tensor = torch.full((x.size(0), 1), fraud_exp, dtype=torch.float32, device=inputs.device)

        # Quantum self‑attention
        rotation_vals = np.concatenate([proj[0], np.zeros(10)])
        entangle_vals = np.zeros(3)
        attn = self.quantum_attention.run(rotation_vals, entangle_vals)
        attn_tensor = torch.tensor(attn, dtype=torch.float32, device=inputs.device).unsqueeze(0).expand(x.size(0), -1)

        # Concatenate and head
        concat = torch.cat((x, fraud_tensor, attn_tensor), dim=-1)
        logits = self.fc3(concat)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "HybridQuantumBinaryClassifier",
]
