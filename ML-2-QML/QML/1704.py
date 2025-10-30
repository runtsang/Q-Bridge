"""
Hybrid classical‑quantum binary classifier.
It replaces the original parametric two‑qubit circuit with a Pennylane
ansatz and implements a parameter‑shift gradient for differentiability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Iterable

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]

class QuantumCircuit:
    """Two‑qubit variational ansatz executed on a Pennylane device."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    def circuit(self, params: Iterable[float]) -> None:
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i, theta in enumerate(params):
            qml.RY(theta, wires=i)

    def expectation(self, params: Iterable[float]) -> float:
        @qml.qnode(self.dev, interface="torch")
        def circuit_fn():
            self.circuit(params)
            return qml.expval(qml.PauliZ(0))
        return circuit_fn()

    def run(self, params: Iterable[float]) -> float:
        return self.expectation(params)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.inputs = inputs.detach().clone()
        exp_vals = []
        for val in ctx.inputs.tolist():
            exp_vals.append(circuit.run([val]))
        return torch.tensor(exp_vals, device=inputs.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = []
        for val in ctx.inputs:
            pos = ctx.circuit.run([val.item() + ctx.shift])
            neg = ctx.circuit.run([val.item() - ctx.shift])
            grad.append((pos - neg) / (2 * ctx.shift))
        grad = torch.tensor(grad, device=ctx.inputs.device, dtype=torch.float32)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a Pennylane circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(-1), self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """CNN followed by a quantum expectation head."""
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
        self.hybrid = Hybrid(n_qubits=2, shots=1024, shift=np.pi / 2)

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
        x = self.fc3(x).squeeze(-1)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def train_one_epoch(self, dataloader: Iterable, optimizer, criterion, device: torch.device):
        self.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
