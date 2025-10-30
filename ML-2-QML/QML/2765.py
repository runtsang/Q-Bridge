"""Quantum implementation of a hybrid autoencoder-based binary classifier.

This module contains:
- QuantumCircuitWrapper that executes a parameterised circuit on Aer.
- QuantumAutoencoder that compresses classical features into a latent vector.
- HybridFunction that bridges PyTorch gradients to the quantum circuit.
- HybridAutoencoderClassifier that mirrors the classical counterpart but replaces the autoencoder
  and classification head with quantum circuits.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.quantum_info import Statevector, Pauli
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------------- #
# Quantum utilities
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """
    Simple wrapper around a parameterised circuit executed on Aer.
    The circuit is expected to have a single measurement on a designated qubit
    whose expectation value is returned.
    """
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.params = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        # Example ansatz: Ry rotations on each qubit
        for i, param in enumerate(self.params):
            self.circuit.ry(param, i)
        self.circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of input angles.
        Returns the expectation value of Z on the first qubit for each sample.
        """
        expectations = []
        for angle_row in angles:
            bound_circuit = self.circuit.bind_parameters(
                {param: val for param, val in zip(self.params, angle_row)}
            )
            state = Statevector.from_instruction(bound_circuit)
            exp = state.expectation_value(Pauli("Z"), [0]).real
            expectations.append(exp)
        return np.array(expectations, dtype=np.float32)

# --------------------------------------------------------------------------- #
# HybridFunction: differentiable interface to the quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Wraps a QuantumCircuitWrapper to be differentiable in PyTorch."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the circuit
        angles = inputs.numpy() + shift
        expectations = circuit.run(angles)
        result = torch.tensor(expectations, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.numpy()):
            # Finite difference with shift
            right = ctx.circuit.run([value + shift[idx]])[0]
            left = ctx.circuit.run([value - shift[idx]])[0]
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

# --------------------------------------------------------------------------- #
# Hybrid layer: forward activations through a quantum circuit
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Accepts batch of scalars or vectors
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# Quantum Autoencoder
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(nn.Module):
    """
    Quantum autoencoder that compresses classical features into a latent vector.
    The encoder is a parameterised circuit; decoding is performed classically.
    """
    def __init__(self, latent_dim: int, backend: AerSimulator, shots: int = 1024) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.backend = backend
        self.shots = shots
        # Encoder circuit: map each input feature to a rotation on a dedicated qubit
        self.circuit = QuantumCircuit(latent_dim)
        self.params = [Parameter(f"theta_{i}") for i in range(latent_dim)]
        for i, param in enumerate(self.params):
            self.circuit.ry(param, i)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        x: batch of feature vectors (batch, latent_dim)
        Returns: encoded latent vectors (batch, latent_dim) as expectation values of Z on each qubit.
        """
        batch_size = x.shape[0]
        expectations = []
        for i in range(batch_size):
            bound = self.circuit.bind_parameters(
                {param: val.item() for param, val in zip(self.params, x[i])}
            )
            state = Statevector.from_instruction(bound)
            exp = [state.expectation_value(Pauli("Z"), [j]).real for j in range(self.latent_dim)]
            expectations.append(exp)
        return torch.tensor(expectations, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# HybridAutoencoderClassifier (Quantum)
# --------------------------------------------------------------------------- #
class HybridAutoencoderClassifier(nn.Module):
    """
    Quantum hybrid classifier mirroring the classical counterpart.
    CNN → FC → Quantum autoencoder → Quantum hybrid classification head.
    """
    def __init__(
        self,
        quantum_backend: AerSimulator = AerSimulator(),
        shots: int = 1024,
    ) -> None:
        super().__init__()
        # Convolutional front‑end
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)

        # Quantum autoencoder
        # Use 84 qubits to match the classical FC output dimension
        self.quantum_autoencoder = QuantumAutoencoder(84, quantum_backend, shots)

        # Quantum classification head
        self.quantum_classifier = Hybrid(
            n_qubits=84,
            backend=quantum_backend,
            shots=shots,
            shift=np.pi / 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        # Flatten and feed into FC backbone
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))

        # Quantum autoencoder encoding
        latent = self.quantum_autoencoder(x)

        # Quantum classification head
        logits = self.quantum_classifier(latent).squeeze(-1)

        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "QuantumCircuitWrapper",
    "HybridFunction",
    "Hybrid",
    "QuantumAutoencoder",
    "HybridAutoencoderClassifier",
]
