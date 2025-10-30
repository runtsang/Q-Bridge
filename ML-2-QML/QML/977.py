"""
Hybrid quantum–classical binary classifier – quantum part.

Implements a two‑qubit variational circuit that receives a single
parameter from the classical head.  The circuit is wrapped in a
PyTorch autograd function so gradients flow through the expectation
value.  The full network contains a ResNet feature extractor, a
linear head that produces a scalar, and the quantum hybrid layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Two‑qubit parametrised circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Build circuit: H on all qubits, RY(theta) on each, CX between qubits,
        # measure all qubits.
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each parameter vector in ``thetas``.

        Parameters
        ----------
        thetas : np.ndarray
            Shape (batch, n_qubits).  Only the first qubit's angle is used
            to drive the circuit; the second qubit receives the same angle
            for simplicity.

        Returns
        -------
        np.ndarray
            Expectation value of Z on qubit 0 for each input.
        """
        compiled = transpile(self.circuit, self.backend)
        expectations = []
        for theta_vec in thetas:
            # Bind the first angle to the circuit parameter.
            bound = {self.theta: theta_vec[0]}
            qobj = assemble(compiled, shots=self.shots,
                            parameter_binds=[bound])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            # Compute expectation of Z on qubit 0.
            exp = 0.0
            for bitstring, count in result.items():
                # Qiskit uses little‑endian ordering; bit 0 is the rightmost.
                bit0 = int(bitstring[0])
                exp += ((-1) ** bit0) * count
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert inputs to numpy array.
        thetas = inputs.detach().cpu().numpy().reshape(-1, 1)
        # Repeat the same angle for the second qubit.
        thetas = np.concatenate([thetas, thetas], axis=1)
        expectations = ctx.circuit.run(thetas)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.detach().cpu().numpy()):
            right = ctx.circuit.run(np.array([[value + shift[idx], value + shift[idx]]]))
            left = ctx.circuit.run(np.array([[value - shift[idx], value - shift[idx]]]))
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=grad_output.dtype, device=grad_output.device)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a scalar through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure scalar input.
        squeezed = torch.squeeze(inputs)
        return HybridFunction.apply(squeezed, self.circuit, self.shift)

# Re‑use the ResNet feature extractor from the classical module.
# The implementation is duplicated here for self‑containment.
class ResidualBlock(nn.Module):
    """A residual block with two 3×3 convolutions."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
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
                nn.BatchNorm2d(out_channels),
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

class ResNetFeatureExtractor(nn.Module):
    """Three‑stage ResNet that reduces spatial size to 1×1."""
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

class HybridClassifier(nn.Module):
    """
    Hybrid quantum‑classical binary classifier that mirrors the
    classical version but replaces the final linear head with a
    variational quantum circuit.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = ResNetFeatureExtractor()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=500, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Pass the scalar through the quantum hybrid layer.
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "ResidualBlock",
           "ResNetFeatureExtractor", "HybridClassifier"]
