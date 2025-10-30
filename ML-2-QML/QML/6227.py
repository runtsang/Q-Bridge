"""Quantum QCNet with ResNet backbone and quantum expectation head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class ResBlock(nn.Module):
    """Basic residual block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class QuantumCircuit(nn.Module):
    """Parameterised two‑qubit circuit that outputs the Z‑expectation of qubit 0."""
    def __init__(self, n_qubits=2, backend=AerSimulator()):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.theta = qiskit.circuit.Parameter('theta')
        self._base = qiskit.QuantumCircuit(n_qubits)
        self._base.h(range(n_qubits))
        self._base.ry(self.theta, 0)
        self._base.measure_all()

    def expectation(self, theta):
        bound = self._base.bind_parameters({self.theta: theta})
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=200)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, cnt in counts.items():
            val = 1 if bitstring[-1] == '0' else -1
            exp += val * cnt
        exp /= 200
        return torch.tensor(exp, dtype=torch.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit via parameter‑shift."""
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        exp_vals = []
        for val in inputs.detach().cpu().numpy():
            exp_vals.append(circuit.expectation(val))
        out = torch.stack(exp_vals).to(inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = torch.zeros_like(inputs)
        for i, val in enumerate(inputs.detach().cpu().numpy()):
            left = ctx.circuit.expectation(val - shift)
            right = ctx.circuit.expectation(val + shift)
            grad = (right - left) / (2 * shift)
            grad_inputs[i] = grad
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum head applying the same circuit to each embedding component."""
    def __init__(self, n_qubits=2, shift=np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits)
        self.shift = shift

    def forward(self, inputs):
        # inputs shape (batch, dim)
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QCNet(nn.Module):
    """ResNet backbone + quantum hybrid head."""
    def __init__(self, num_classes=2, n_qubits=2, shift=np.pi/2):
        super().__init__()
        # Encoder identical to the classical version but without classifier
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hybrid = Hybrid(n_qubits, shift)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = x  # raw features before quantum head
        q = self.hybrid(embedding)
        logits = self.classifier(q)
        probs = F.softmax(logits, dim=-1)
        return probs, embedding

__all__ = ["QCNet"]
