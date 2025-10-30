from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """
    Simple two‑qubit parameterised circuit executed on the Aer simulator.
    Mirrors the QuantumCircuit wrapper from the original QML seed.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple entangling pattern
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
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
            probs = counts / self.shots
            # Expectation of Pauli‑Z: |0> → +1, |1> → −1
            return np.sum(np.array(list(count_dict.keys()), dtype=int) * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit via the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        thetas = inputs.detach().cpu().numpy().flatten().tolist()
        expectation = circuit.run(thetas)
        return torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grad_inputs = []
        for val in inputs.detach().cpu().numpy().flatten():
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grad_inputs.append((right - left) / 2.0)
        grad_tensor = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a parameterised quantum circuit.
    """
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """
    CNN + Quantum expectation head classifier that mirrors the classical hybrid model.
    The API is identical to HybridBinaryClassifier in the classical module.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        n_qubits: int = 2,
        shift: float = np.pi / 2,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected block
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = Hybrid(n_qubits, shift=shift, shots=shots)

        # Final classification head
        self.head = nn.Linear(1, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

        q_out = self.hybrid(x)
        logits = self.head(q_out)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridBinaryClassifier", "QuantumCircuitWrapper", "HybridFunction", "Hybrid"]
