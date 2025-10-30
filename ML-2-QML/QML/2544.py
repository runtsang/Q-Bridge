"""Hybrid kernel‑based binary classifier with a quantum expectation head.

The model retains the same CNN backbone as the classical version but replaces
the linear sigmoid head with a parameterised quantum circuit that evaluates
the expectation value of a Z‑observable.  The kernel embedding is computed
using the TorchQuantum ansatz, enabling a fully quantum‑enhanced decision
boundary while keeping the overall API identical to the classical counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

# Import the quantum kernel implementation from the shared module
from QuantumKernelMethod import Kernel


class QuantumCircuit:
    """Two‑qubit parametrised circuit executed on the Aer simulator."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridQuantumHead(nn.Module):
    """Quantum expectation head that maps kernel embeddings to a probability."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float = 0.0) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, n_qubits)
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        expectation = self.quantum_circuit.run(thetas)
        probs = torch.sigmoid(torch.tensor(expectation, device=inputs.device) + self.shift)
        return probs


class HybridKernelQCNet(nn.Module):
    """CNN → quantum kernel embedding → quantum expectation head."""

    def __init__(
        self,
        n_support: int = 16,
        n_qubits: int = 4,
        backend=None,
        shots: int = 100,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.kernel = Kernel()
        self.register_buffer("support", torch.randn(n_support, 1))

        self.hybrid_head = HybridQuantumHead(n_support, backend, shots, shift)

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
        x = self.fc3(x)

        k = torch.stack([self.kernel(x, sv) for sv in self.support], dim=1)
        probs = self.hybrid_head(k)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridKernelQCNet"]
