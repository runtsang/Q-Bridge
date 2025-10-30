"""HybridQuantumClassifier – quantum head with tunable depth and basis.

The quantum component is a parameterised variational circuit that
can be configured with a different number of qubits, rotation
depth, and measurement basis.  The circuit is wrapped in a
PyTorch autograd function so that gradients flow through the
quantum expectation value.  The classical backbone remains
unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator


class ParametricQuantumCircuit:
    """A flexible variational circuit on n qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of repeated rotation‑entanglement blocks.
    basis : str
        Measurement basis: ``'z'`` (default) or ``'x'``.
    """

    def __init__(self, n_qubits: int, depth: int = 2, basis: str = "z") -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.basis = basis.lower()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = [Parameter(f"θ_{i}") for i in range(n_qubits * depth)]

        for d in range(depth):
            # Layer of single‑qubit rotations
            for q in range(n_qubits):
                self.circuit.ry(self.params[d * n_qubits + q], q)
            # Entangling layer
            for q in range(n_qubits - 1):
                self.circuit.cx(q, q + 1)
            if n_qubits > 1:
                self.circuit.cx(n_qubits - 1, 0)

        # Measurement
        if self.basis == "x":
            for q in range(n_qubits):
                self.circuit.h(q)
        self.circuit.measure_all()

        self.backend = AerSimulator()
        self.shots = 1024

    def run(self, angles: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angle vectors."""
        if angles.ndim == 1:
            angles = angles[None, :]
        bound_circuits = [
            self.circuit.bind_parameters(dict(zip(self.params, ang)))
            for ang in angles
        ]
        compiled = transpile(bound_circuits, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()

        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=int)
            return np.sum(states * probs) / (2 ** self.n_qubits - 1)

        expectations = []
        for circuit in result.get_counts():
            expectations.append(expectation(circuit))
        return np.array(expectations)

class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: ParametricQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # The expectation value is a scalar per batch element
        exp_vals = circuit.run(inputs.tolist())
        ctx.save_for_backward(inputs)
        return torch.tensor(exp_vals, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.tolist():
            # Parameter shift rule
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grad_inputs.append((right - left) / 2)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad_inputs * grad_output, None, None

class QuantumHybrid(nn.Module):
    """Hybrid layer that forwards activations through a variational circuit."""

    def __init__(self, n_qubits: int, depth: int = 2, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = ParametricQuantumCircuit(n_qubits, depth=depth)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return QuantumHybridFunction.apply(squeezed, self.circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""

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
        self.hybrid = QuantumHybrid(n_qubits=2, depth=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["ParametricQuantumCircuit", "QuantumHybridFunction",
           "QuantumHybrid", "QCNet"]
