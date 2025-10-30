"""Quantum hybrid network with variational circuit head and noise model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

class VariationalQuantumCircuit:
    """2‑qubit variational circuit with Ry and Rz layers and CNOT entanglement."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [f"theta_{i}" for i in range(n_qubits * 2)]
        for i in range(n_qubits):
            self.circuit.ry(0.0, i)
            self.circuit.rz(0.0, i)
        self.circuit.barrier()
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        for i in range(n_qubits):
            self.circuit.ry(0.0, i)
            self.circuit.rz(0.0, i)
        self.circuit.measure_all()

    def bind_parameters(self, params: np.ndarray) -> QuantumCircuit:
        param_dict = {self.theta[i]: params[i] for i in range(len(self.theta))}
        return self.circuit.bind_parameters(param_dict)

    def expectation(self, params: np.ndarray, backend: AerSimulator) -> np.ndarray:
        bound_circ = self.bind_parameters(params)
        transpiled = transpile(bound_circ, backend=backend)
        qobj = assemble(transpiled, shots=self.shots)
        result = backend.run(qobj).result()
        counts = result.get_counts()
        exp = []
        for i in range(self.n_qubits):
            p0 = 0
            p1 = 0
            for state, cnt in counts.items():
                if state[self.n_qubits - 1 - i] == '0':
                    p0 += cnt
                else:
                    p1 += cnt
            exp.append((p0 - p1) / self.shots)
        return np.array(exp)

class QuantumHybridLayer(nn.Module):
    """Differentiable layer that forwards through a variational quantum circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024, noise: bool = False, depolarizing_strength: float = 0.01):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        if noise:
            error = depolarizing_error(depolarizing_strength, 1)
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz', 'cx'])
            self.backend.set_options(noise_model=noise_model)
        self.quantum_circuit = VariationalQuantumCircuit(n_qubits, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            params = x[i].detach().cpu().numpy()
            exp = self.quantum_circuit.expectation(params, self.backend)
            out.append(exp)
        return torch.tensor(out, dtype=torch.float32, device=x.device)

class QuantumHybridClassifier(nn.Module):
    """CNN followed by quantum hybrid layer."""
    def __init__(self, num_classes: int = 2, noise: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.quantum = QuantumHybridLayer(n_qubits=num_classes, shots=1024, noise=noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        return self.quantum(x)

__all__ = ["VariationalQuantumCircuit", "QuantumHybridLayer", "QuantumHybridClassifier"]
