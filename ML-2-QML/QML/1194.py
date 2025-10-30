"""
HybridBinaryClassifier – Quantum‑aware model with a variational circuit head.

The network mirrors the classical counterpart but replaces the sigmoid head with a
parameterised two‑qubit variational circuit.  A learnable 1×1 convolution acts as a
feature‑map before the dense layers.  The quantum circuit is executed on Qiskit’s
Aer simulator and is wrapped in a custom autograd function that implements the
parameter‑shift rule for gradients.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit as QiskitCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """Two‑qubit variational circuit with parameterised RY gates."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = QiskitCircuit.Parameter("theta")
        self.circuit = QiskitCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, 0)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the given angles and return expectation of Z."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Wraps the quantum circuit and implements the shift‑rule gradient."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # inputs is a 1‑D tensor of shape (batch,)
        thetas = inputs.detach().cpu().numpy()
        exp_values = circuit.run(thetas)
        outputs = torch.tensor(exp_values, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        grad_inputs = []
        for theta in thetas:
            exp_plus = ctx.circuit.run([theta + shift])[0]
            exp_minus = ctx.circuit.run([theta - shift])[0]
            grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum head that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int, shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift
        backend = AerSimulator()
        self.circuit = VariationalQuantumCircuit(n_qubits, backend, shots=1024)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN with a feature‑map, dense layers, and a variational quantum head."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Conv2d(3, 32, kernel_size=1)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        dummy_input = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            x = self.feature_map(dummy_input)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = self.drop1(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.drop1(x)
            x = torch.flatten(x, 1)
        self.fc1 = nn.Linear(x.size(1), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=2, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["VariationalQuantumCircuit", "HybridFunction", "Hybrid", "HybridBinaryClassifier"]
