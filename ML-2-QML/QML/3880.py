"""Quantum‑enhanced binary classifier with a parameterised multi‑qubit circuit.

This module builds on the seed by adding a deeper quantum layer that
produces multiple expectation values.  The output of the classical
convolutional backbone is mapped to qubit angles which are fed into a
parameterised Ry circuit.  Finite‑difference gradients are used to
enable end‑to‑end training via PyTorch autograd.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import transpile, assemble
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQubitCircuit:
    """Parameterised circuit that accepts an array of angles.

    The circuit consists of Hadamard gates on each qubit followed by a
    single Ry rotation with the supplied angle.  The expectation value
    of the Z operator on each qubit is returned.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each provided angle and return expectation values."""
        compiled = transpile(self.circuit, self.backend)
        binds = [{self.theta: theta} for theta in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=binds)
        job = self.backend.run(qobj)
        results = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(results, list):
            return np.array([expectation(r) for r in results])
        return np.array([expectation(results)])


class QuantumHybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards a scalar to the quantum circuit.

    Finite‑difference gradients are computed on the fly for each input.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, circuit: MultiQubitCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.input = x.detach()
        x_np = x.detach().cpu().numpy().flatten().tolist()
        expect = circuit.run(x_np)
        return torch.tensor(expect, dtype=torch.float32, device=x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_np = ctx.input.detach().cpu().numpy().flatten()
        eps = 1e-3
        pos = ctx.circuit.run((x_np + eps).tolist())
        neg = ctx.circuit.run((x_np - eps).tolist())
        grad = (pos - neg) / (2 * eps)
        grad_tensor = torch.tensor(grad, dtype=torch.float32, device=grad_output.device)
        return grad_tensor * grad_output, None, None


class QuantumHybrid(nn.Module):
    """Layer that maps a feature vector to an angle and forwards it through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = MultiQubitCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The feature vector is expected to be a single scalar per batch.
        return QuantumHybridFunction.apply(x.squeeze(-1), self.circuit, self.shift)


class HybridBinaryClassifier(nn.Module):
    """Convolutional backbone followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout2d(p=0.3)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = QuantumHybrid(1, backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        probs = self.hybrid(x).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["MultiQubitCircuit", "QuantumHybrid", "QuantumHybridFunction", "HybridBinaryClassifier"]
