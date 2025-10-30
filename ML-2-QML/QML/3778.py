"""Hybrid quantum‑classical binary classifier with Qiskit backend and fraud‑style scaling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Simple two‑qubit variational circuit with a single rotation parameter."""
    def __init__(self, n_qubits: int, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
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

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to a variational quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.cpu().numpy().tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.cpu().numpy():
            exp_plus = ctx.circuit.run([val + shift].tolist())
            exp_minus = ctx.circuit.run([val - shift].tolist())
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit and applies learnable scale/shift."""
    def __init__(self, n_qubits: int, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, shots)
        self.shift = shift
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q_out = HybridFunction.apply(
            torch.squeeze(inputs, dim=-1),
            self.quantum_circuit,
            self.shift,
        )
        return self.scale * q_out + self.bias

class HybridBinaryClassifier(nn.Module):
    """Quantum‑enhanced binary classifier with a convolutional backbone and a variational head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=2, shots=100, shift=np.pi / 2)

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
        logits = self.hybrid(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "HybridBinaryClassifier"]
