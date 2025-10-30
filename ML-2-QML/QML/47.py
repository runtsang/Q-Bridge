"""HybridQuantumClassifier: classical backbone + quantum head for binary classification.

This module extends the original seed by adding a parameterised two‑qubit
ansatz with configurable depth. The quantum head is implemented via a
parameter‑shift analytic gradient and runs on the Aer simulator. The
backbone is identical to the classical version so that the two modules
share the same API.

API:
    HybridQuantumClassifier(num_classes=2, device='cpu')
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

__all__ = ["HybridQuantumClassifier", "ResidualBlock", "QuantumCircuit", "Hybrid", "HybridFunction"]

class ResidualBlock(nn.Module):
    """Simple 2‑D residual block with batch‑norm."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class QuantumCircuit:
    """Parameterized two‑qubit variational circuit."""
    def __init__(self, n_qubits: int = 2, depth: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = qiskit.circuit.ParameterVector("theta", self.n_qubits * self.depth)
        self._build_circuit()

    def _build_circuit(self):
        for d in range(self.depth):
            for q in range(self.n_qubits):
                self.circuit.ry(self.theta[d * self.n_qubits + q], q)
            # entangling layer
            self.circuit.cx(0, 1)
            self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for an array of parameter sets."""
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        results = []
        for params in thetas:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots,
                            parameter_binds=[{self.theta[i]: params[i] for i in range(len(params))}])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            # Expectation of Z on first qubit
            exp_z = 0.0
            for state, count in result.items():
                z = 1 if state[0] == '0' else -1
                exp_z += z * count / self.shots
            results.append(exp_z)
        return np.array(results)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs
        thetas = inputs.detach().cpu().numpy()
        exp_z = circuit.run(thetas)
        return torch.tensor(exp_z, device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs = ctx.inputs
        grad_inputs = []
        for idx in range(inputs.shape[0]):
            theta = inputs[idx].item()
            exp_plus = circuit.run(np.array([theta + shift]))
            exp_minus = circuit.run(np.array([theta - shift]))
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_inputs.append(grad)
        grad_tensor = torch.tensor(grad_inputs, device=grad_output.device, dtype=grad_output.dtype)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, depth: int, shift: float = math.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, depth)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(HybridFunction.apply(x.squeeze(), self.circuit, self.shift))

class HybridQuantumClassifier(nn.Module):
    """Classical backbone + quantum head for binary classification."""
    def __init__(self, num_classes: int = 2, device: str = "cpu", depth: int = 2):
        super().__init__()
        self.device = device
        # Backbone identical to classical version
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 8 * 8, 1)
        self.quantum_head = Hybrid(1, depth=depth, shift=math.pi / 2)
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        probs = self.quantum_head(x)
        return probs

    def compute_loss(self, probs: torch.Tensor, targets: torch.Tensor,
                     use_focal: bool = False, gamma: float = 2.0) -> torch.Tensor:
        if use_focal:
            focal = FocalLoss(gamma=gamma)
            return focal(probs, targets)
        else:
            return self.loss_fn(probs, targets)
