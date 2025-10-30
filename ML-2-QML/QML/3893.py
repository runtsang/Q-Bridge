"""HybridSamplerNet – quantum‑enabled implementation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_aer import AerSimulator


class QuantumSamplerCircuit:
    """Parameterised quantum circuit that acts as a sampler."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.input_params = ParameterVector('input', n_qubits)
        self.weight_params = ParameterVector('weight', 4)

        # Example circuit: two‑qubit Ry rotations, CX, then trainable Ry layers
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Sampler primitive
        self.sampler = QiskitSampler(backend=self.backend, shots=self.shots)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run the circuit for a batch of input angles and return probabilities."""
        param_dicts = [
            {self.input_params[0]: inp[0], self.input_params[1]: inp[1]}
            for inp in inputs
        ]
        result = self.sampler.run(self.circuit, param_dicts)
        probs = result.get_counts()
        # Convert counts to probabilities for each basis state
        probs_array = np.zeros((len(inputs), 2 ** self.n_qubits))
        for idx, count_dict in enumerate(probs):
            for state, count in count_dict.items():
                probs_array[idx, int(state, 2)] = count / self.shots
        return probs_array


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum hybrid head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumSamplerCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit

        # Forward pass: evaluate expectation of Z on first qubit
        exp_expectation = circuit.run(inputs.reshape(-1, 1).tolist())
        # Take expectation of Pauli Z on qubit 0
        probs = exp_expectation[:, 0]
        result = torch.tensor(probs, dtype=torch.float32).unsqueeze(-1)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.tolist()):
            right = ctx.circuit.run([value + shift[idx]])
            left = ctx.circuit.run([value - shift[idx]])
            gradients.append(right[0, 0] - left[0, 0])
        gradients = torch.tensor(gradients, dtype=torch.float32).unsqueeze(-1)
        return gradients * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumSamplerCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.circuit, self.shift)


class HybridSamplerNet(nn.Module):
    """Quantum CNN → Hybrid head → Quantum sampler → binary probabilities."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head and sampler
        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=256, shift=np.pi / 2)
        # The sampler is a quantum statevector sampler that outputs probability amplitudes
        self.sampler = self.hybrid.circuit.sampler

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
        x = self.hybrid(x).T  # shape (batch, 1)
        # Prepare inputs for the sampler: each sample is a 2‑dim vector
        sampler_inputs = torch.cat([x, torch.zeros_like(x)], dim=-1)
        # Run quantum sampler to obtain probabilities for |00> and |01>
        probs = self.sampler.run(sampler_inputs.tolist())
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        return torch.cat((probs_tensor, 1 - probs_tensor), dim=-1)


__all__ = ["QuantumSamplerCircuit", "HybridFunction", "Hybrid", "HybridSamplerNet"]
