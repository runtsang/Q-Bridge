"""Hybrid quantum kernel module with Qiskit and a quantum‑classical classifier.

The module exposes:
- QuantumCircuit: a parameterised circuit that can be executed on a simulator.
- HybridFunction: differentiable bridge between PyTorch and the circuit.
- Hybrid: a wrapper that passes activations through the quantum circuit.
- QCNet: convolutional backbone followed by the quantum expectation head.

This mirrors the original `QuantumKernelMethod.py` but augments it with
a trainable Ry gate and a random layer to increase expressivity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit.library import RandomQuantumCircuit


class QuantumCircuit:
    """Wrapper around a parametrised random circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        # Random layer of depth 3 to increase expressivity
        self._circuit = RandomQuantumCircuit(n_qubits, depth=3)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        # Add a global Ry rotation that will be treated as the learnable
        # parameter in the differentiable wrapper.
        for q in all_qubits:
            self._circuit.ry(self.theta, q)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the parametrised circuit for the provided angles."""
        compiled = transpile(self._circuit, self.backend)
        param_binds = [{self.theta: theta} for theta in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.circuit.run([value + shift[idx]])
            expectation_left = ctx.circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


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

        backend = qiskit.Aer.get_backend("aer_simulator")
        # The number of qubits matches the output dimension of the last fully
        # connected layer; we use 4 qubits here for brevity.
        self.hybrid = Hybrid(n_qubits=4, backend=backend, shots=200, shift=np.pi / 2)

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
        # Quantum expectation head
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
