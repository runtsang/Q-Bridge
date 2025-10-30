"""HybridAdvancedClassifier
========================

Quantum‑classical binary classifier that extends the original
QCNet with a 3‑qubit parameter‑shift circuit and a differentiable
expectation wrapper. The classical backbone is identical to the
classical version but the head is replaced with a quantum layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

class ResidualDenseBlock(nn.Module):
    """Residual block with dense connections across layers."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = F.relu(out)
        out = self.dropout(out)
        return out

class DifferentiableQuantumExpectation(torch.autograd.Function):
    """
    Differentiable wrapper around a parameter‑shift quantum circuit.
    The forward pass evaluates the circuit for the supplied angles,
    while the backward pass uses the parameter‑shift rule to compute
    gradients analytically.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy().flatten()
        expectation = circuit.run(angles)
        output = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        angles = inputs.detach().cpu().numpy().flatten()
        grads = []
        for theta in angles:
            plus = circuit.run([theta + shift])
            minus = circuit.run([theta - shift])
            grad = (plus - minus) / (2 * shift)
            grads.append(grad)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class QuantumCircuit:
    """
    3‑qubit parameter‑shift circuit implemented with Qiskit.
    The circuit prepares a GHZ‑like state, applies RY rotations
    with the input angles, and measures the expectation of Z⊗Z⊗Z.
    """
    def __init__(self, backend=AerSimulator(), shots: int = 500):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(3)
        self.theta = qiskit.circuit.Parameter('theta')
        # GHZ preparation
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.cx(0, 2)
        # Parameterised rotations
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.ry(self.theta, 2)
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        results = []
        for theta in thetas:
            bound = compiled.bind_parameters({self.theta: theta})
            qobj = assemble(bound, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            expectation = 0.0
            for bitstring, count in counts.items():
                z = np.array([1 if b == '0' else -1 for b in bitstring])
                exp_val = np.prod(z)
                expectation += exp_val * count
            expectation /= self.shots
            results.append(expectation)
        return np.array(results)

class HybridAdvancedClassifier(nn.Module):
    """
    Hybrid neural network with a quantum expectation head.
    The classical backbone mirrors the one used in the
    classical variant. The head applies a parameter‑shifted
    3‑qubit circuit followed by a sigmoid to obtain class
    probabilities.
    """
    def __init__(self, dropout_rate: float = 0.5, shots: int = 500):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.rd_block = ResidualDenseBlock(6, 15)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.quantum = QuantumCircuit(shots=shots)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor, mc_samples: int = 1) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.rd_block(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if mc_samples > 1:
            preds = []
            self.train()
            for _ in range(mc_samples):
                y = self.dropout(x)
                q_exp = DifferentiableQuantumExpectation.apply(y, self.quantum, self.shift)
                prob = torch.sigmoid(q_exp)
                preds.append(prob)
            probs = torch.stack(preds, dim=0).mean(0)
            self.eval()
        else:
            y = self.dropout(x)
            q_exp = DifferentiableQuantumExpectation.apply(y, self.quantum, self.shift)
            probs = torch.sigmoid(q_exp)
        return torch.cat((probs, 1 - probs), dim=-1)

    def get_mc_predictions(self, x: torch.Tensor, mc_samples: int = 10) -> torch.Tensor:
        return self.forward(x, mc_samples=mc_samples)

__all__ = ["HybridAdvancedClassifier", "ResidualDenseBlock", "DifferentiableQuantumExpectation", "QuantumCircuit"]
