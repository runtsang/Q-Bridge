"""Hybrid quantum‑classical network with a 4‑qubit variational circuit head.

The implementation follows the same public API as its classical counterpart
(`HybridQCNet`) but replaces the dense head with a parameter‑shift differentiable
quantum circuit.  The circuit is executed on the Aer simulator and returns
the expectation of the first qubit in the computational basis.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuit:
    """
    Parameterised 4‑qubit circuit executed on Aer.  The ansatz consists of
    a layer of Hadamards, a chain of CX gates, and a parametrised Ry rotation
    on each qubit.  The circuit measures all qubits and the expectation of the
    first qubit (Z observable) is returned.
    """

    def __init__(self, n_qubits: int, backend: qiskit.providers.Provider, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def expectation(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each theta in *thetas* and return the
        expectation value of the first qubit in the Z basis.
        """
        exps = []
        for theta in np.atleast_1d(thetas):
            compiled = transpile(self.circuit, self.backend, optimization_level=2)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: theta}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(s[::-1], 2) for s in counts.keys()])
            # expectation of Z on first qubit: +1 for |0>, -1 for |1>
            exp = np.sum((1 - 2 * (states >> (self.n_qubits - 1) & 1)) * probs)
            exps.append(exp)
        return np.array(exps)


class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards a scalar through a quantum circuit
    and applies the parameter‑shift rule to compute gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        # Convert to numpy for circuit execution
        thetas = inputs.detach().cpu().numpy().astype(float)
        exps = circuit.expectation(thetas)
        out = torch.tensor(exps, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy().astype(float)
        grads = []
        for theta in thetas:
            right = ctx.circuit.expectation(theta + shift)
            left = ctx.circuit.expectation(theta - shift)
            grads.append((right - left) / (2 * shift))
        grad_tensor = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_tensor * grad_output, None, None


class Hybrid(nn.Module):
    """
    Module that forwards a scalar through the quantum circuit using the
    differentiable `HybridFunction`.
    """

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 512, shift: float = np.pi / 2) -> None:
        super().__init__()
        if backend is None:
            backend = AerSimulator()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x)
        return HybridFunction.apply(x, self.circuit, self.shift)


class HybridQCNet(nn.Module):
    """
    Hybrid quantum‑classical network that mirrors the classical architecture
    while using a variational circuit as the head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits=4, backend=backend, shots=512, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


__all__ = ["HybridQCNet"]
