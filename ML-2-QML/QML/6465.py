"""Quantum counterpart of the hybrid QCNN classifier.

This module builds a quantum expectation head using a single‑qubit
parameter‑shift circuit. The classical convolutional backbone mirrors
the structure of the classical model, and the quantum head is fully
differentiable via a custom autograd function.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Simple two‑qubit circuit with a single tunable parameter."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        self.backend = AerSimulator()
        self.shots = shots
        self.n_qubits = n_qubits
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised Ry on each qubit
        theta = qc._parameter("theta")
        qc.h(0)
        qc.h(1)
        qc.ry(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit with the provided parameters."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[0]: params[0]}]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of Z on qubit 0
        probs = np.array(list(result.values())) / self.shots
        states = np.array([int(k[0]) for k in result.keys()])
        return np.sum(states * probs)

class HybridFunction(torch.autograd.Function):
    """PyTorch autograd wrapper for the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to numpy
        params = inputs.detach().cpu().numpy()
        expectation = circuit.run(params)
        result = torch.tensor([expectation], dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.cpu().numpy(), ctx.shift)
        grads = []
        for idx, val in enumerate(inputs.cpu().numpy()):
            right = ctx.circuit.run([val + shift[idx]])
            left = ctx.circuit.run([val - shift[idx]])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Layer that forwards a scalar through the quantum circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs)
        return HybridFunction.apply(squeezed, self.circuit, self.shift)

class HybridQCNNClassifier(nn.Module):
    """CNN backbone followed by a QCNN‑style quantum head."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        super().__init__()
        # Classical convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum head
        self.hybrid = Hybrid(n_qubits, shots)

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
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["HybridQCNNClassifier", "Hybrid", "HybridFunction", "QuantumCircuitWrapper"]
