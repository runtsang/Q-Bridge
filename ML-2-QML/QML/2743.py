"""Hybrid quantum‑classical binary classifier with a differentiable two‑qubit circuit.

The quantum head is a parameterised 2‑qubit circuit built from H, RY, CX, and RZ gates.
It is wrapped in a PyTorch autograd function that uses the shift rule for gradients.
The circuit is executed on Qiskit Aer and can be trained jointly with a classical CNN backbone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parameterized 2‑qubit circuit with H, RY, CX, and RZ gates."""
    def __init__(self, backend: qiskit.providers.Backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, 0)
        self.circuit.cx(0, 1)
        self.circuit.rz(self.theta, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        def expectation(count_dict):
            probs = {k: v / self.shots for k, v in count_dict.items()}
            # Expectation of Z on qubit 0: +1 for |0>, -1 for |1>
            return probs.get('0', 0) * 1 + probs.get('1', 0) * (-1)
        if isinstance(result, list):
            return np.array([expectation(c) for c in result])
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    """Forward pass uses quantum expectation; backward uses shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        expectation = circuit.run(thetas)
        out = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for val in inputs.detach().cpu().numpy().flatten():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grad.append((right - left) / 2)
        grad = torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a scalar through the quantum circuit."""
    def __init__(self, backend: qiskit.providers.Backend = AerSimulator(), shots: int = 512, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridClassifier(nn.Module):
    """CNN backbone followed by a quantum head."""
    def __init__(self, backend: qiskit.providers.Backend = AerSimulator()):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),  # two logits
        )
        self.hybrid = Hybrid(backend, shots=512, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        scalar = logits.mean(dim=1, keepdim=True)
        out = self.hybrid(scalar)
        prob = torch.sigmoid(out)
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["QuantumCircuit", "Hybrid", "HybridFunction", "HybridClassifier"]
