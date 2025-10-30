import numpy as np
import torch
import torch.nn as nn
import math
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Parameterized ansatz with multiple layers of Ry and CX."""
    def __init__(self, n_qubits: int, backend, shots: int, depth: int = 2):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.depth = depth
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.thetas = qiskit.circuit.ParameterVector("theta", n_qubits * depth)
        self.circuit.h(range(n_qubits))
        idx = 0
        for _ in range(depth):
            for q in range(n_qubits):
                self.circuit.ry(self.thetas[idx], q)
                idx += 1
            for q in range(n_qubits - 1):
                self.circuit.cx(q, q + 1)
        self.circuit.measure_all()
    def run(self, angles: np.ndarray) -> np.ndarray:
        results = []
        for angle in angles:
            param_dict = {self.thetas[i]: angle[i] for i in range(len(self.thetas))}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots,
                            parameter_binds=[param_dict])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                bit = state[-1]
                z = 1.0 if bit == '0' else -1.0
                exp += z * cnt / self.shots
            results.append(exp)
        return np.array(results)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        exp = ctx.circuit.run(angles)
        result = torch.tensor(exp, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.detach().cpu().numpy()
        grads = []
        for angle in angles:
            right = ctx.circuit.run(angle + shift)
            left = ctx.circuit.run(angle - shift)
            grads.append((right - left) / 2.0)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum expectation head with learnable angle mapping."""
    def __init__(self, in_features: int, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots, depth=2)
        self.shift = shift
        self.linear = nn.Linear(in_features, n_qubits)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.linear(x)
        return HybridFunction.apply(angles, self.quantum_circuit, self.shift)

class ResidualBlock(nn.Module):
    """Simple 3x3 residual block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SimpleResNetBackbone(nn.Module):
    """Light‑weight ResNet backbone identical to the classical version."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class HybridClassifier(nn.Module):
    """Hybrid ResNet‑based binary classifier with a variational quantum head."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = SimpleResNetBackbone()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(128, 4, backend, shots=200, shift=math.pi / 2)
        self.calibration = nn.Linear(1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.hybrid(x)
        x = self.calibration(x)
        probs = torch.sigmoid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridClassifier"]
