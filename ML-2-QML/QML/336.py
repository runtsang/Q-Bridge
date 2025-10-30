"""Hybrid quantum‑classical binary classifier using a 4‑qubit variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter, Gate
from qiskit.quantum_info import Pauli


class VariationalCircuit:
    """Four‑qubit parametrised circuit with entanglement and measurement of the first qubit."""

    def __init__(self, shots: int = 1024) -> None:
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = Parameter("θ")
        self.circuit = QC(4)
        # Apply entangling layer
        self.circuit.h([0, 1, 2, 3])
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        # Parameterised rotations on each qubit
        for qubit in range(4):
            self.circuit.ry(self.theta, qubit)
        # Measure the first qubit
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles and return the expectation of Z on qubit 0."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts: dict) -> float:
            """Expectation value of Z on qubit 0."""
            # Convert bitstring to integer; the first bit corresponds to qubit 0
            exp = 0.0
            for bitstring, count in counts.items():
                # Qiskit orders bits from most significant (qubit 0) to least
                z = 1 if bitstring[0] == "0" else -1
                exp += z * count
            return exp / self.shots

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the variational circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Forward pass: expectation value for each input angle
        thetas = inputs.detach().cpu().numpy().flatten()
        expectation = circuit.run(thetas)
        # Convert to tensor
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        # Parameter‑shift rule for each input
        for val in inputs.detach().cpu().numpy().flatten():
            right = circuit.run([val + shift])
            left = circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Layer that forwards activations through a variational quantum circuit."""

    def __init__(self, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = VariationalCircuit(shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure shape (batch, 1)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(-1)
        return HybridFunction.apply(inputs.squeeze(-1), self.circuit, self.shift)


class QCNet(nn.Module):
    """CNN followed by a quantum expectation head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.resblock = BasicBlock(6, 15, stride=2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Compute flattened size
        dummy = torch.zeros(1, 3, 32, 32)
        x = self._forward_features(dummy)
        self.fc1 = nn.Linear(x.shape[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(shots=512, shift=np.pi / 2)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.resblock(x)
        x = self.pool(x)
        x = self.drop2(x)
        return torch.flatten(x, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._forward_features(inputs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Quantum head expects shape (batch, 1)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)


# BasicBlock is reused from the classical module; we duplicate it here for self‑containment.
class BasicBlock(nn.Module):
    """Simplified ResNet block used by both classical and quantum nets."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (stride!= 1 or in_channels!= out_channels) and nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


__all__ = ["VariationalCircuit", "HybridFunction", "Hybrid", "BasicBlock", "QCNet"]
