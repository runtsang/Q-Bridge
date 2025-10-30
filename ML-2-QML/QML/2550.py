"""
HybridBinaryClassifier – Quantum‑parameterised PyTorch implementation.

This module implements the same convolutional backbone as the classical
counterpart but replaces the dense head with a parameterised quantum
circuit.  The quantum head uses a shift‑rule differentiable expectation
and can be instantiated as a single‑qubit FCL for lightweight experiments.
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
    Parameterised two‑qubit circuit used as the quantum head.

    The circuit applies a global Hadamard, a rotation Ry(θ) on each qubit,
    and measures all qubits.  The expectation value of the bit‑string
    interpreted as a binary number is returned.
    """

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        return self._expectation(result)

    def _expectation(self, result: dict) -> np.ndarray:
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


class HybridFunction(torch.autograd.Function):
    """
    Differentiable interface between PyTorch and the quantum circuit.

    The forward pass evaluates the quantum circuit.  The backward pass
    uses the central‑difference shift rule to approximate gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Expectation values are computed for each input element
        expectations = circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        # Central‑difference approximation
        right = ctx.circuit.run((inputs + shift).detach().cpu().numpy())
        left = ctx.circuit.run((inputs - shift).detach().cpu().numpy())
        grad = (right - left) / (2 * shift)
        return grad * grad_output, None, None


class QuantumHybridHead(nn.Module):
    """
    Quantum head that forwards activations through a parameterised circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Backend
        Backend used for execution.
    shots : int
        Number of shots per evaluation.
    shift : float
        Shift value for the central‑difference rule.
    """

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Flatten to 1‑D for the quantum circuit
        flat_inputs = inputs.view(-1)
        return HybridFunction.apply(flat_inputs, self.circuit, self.shift)


class HybridBinaryClassifier(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.

    The architecture mirrors the classical version but replaces the final
    dense layer with a quantum circuit.  The head can be swapped with a
    single‑qubit FCL for rapid prototyping.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Iterable[int] = (120, 84),
        dropout: Iterable[float] = (0.2, 0.5),
        n_qubits: int = 2,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=dropout[0])
        self.drop2 = nn.Dropout2d(p=dropout[1])
        self.fc1 = nn.Linear(55815, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        backend = AerSimulator()
        self.head = QuantumHybridHead(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
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
        prob = self.head(x)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "QuantumHybridHead", "HybridBinaryClassifier"]
