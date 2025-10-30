"""Hybrid classical‑quantum convolutional network for binary classification.

This module defines a variational quantum circuit that is integrated
into a PyTorch model via a custom autograd function.  The circuit uses
a simple two‑qubit ansatz and a parameter‑shift rule for gradients.
The classical backbone is identical to the one used in the
`HybridClassicalHead` module, ensuring a fair comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import Aer, transpile, assemble


class ResidualBlock(nn.Module):
    """Lightweight residual block identical to the classical backbone."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class QuantumCircuit:
    """Parametrised two‑qubit variational circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit = self._build_ansatz()

    def _build_ansatz(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(self.theta, i)
            qc.rz(self.theta, i)
            if i < self.n_qubits - 1:
                qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles and return expectation values."""
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: theta} for theta in thetas]
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts_list = result.get_counts()

        def expectation(counts):
            total = 0
            for bitstring, cnt in counts.items():
                total += (1 if bitstring[-1] == "0" else -1) * cnt
            return total / self.shots

        if isinstance(counts_list, list):
            return np.array([expectation(c) for c in counts_list])
        else:
            return np.array([expectation(counts_list)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = ctx.circuit.run(inputs.cpu().numpy())
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        inputs_np = inputs.detach().cpu().numpy()
        grads = []
        for val in inputs_np:
            e_plus = ctx.circuit.run([val + shift])[0]
            e_minus = ctx.circuit.run([val - shift])[0]
            grads.append((e_plus - e_minus) / 2)
        grad_inputs = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a variational circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x.view(-1), self.circuit, self.shift)


class QCNet(nn.Module):
    """Convolutional backbone followed by a quantum expectation head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.resblock = ResidualBlock(6, 12, stride=1)
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.resblock(x)
        x = torch.flatten(x, 1)
        logits = self.hybrid(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
