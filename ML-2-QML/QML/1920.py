"""Quantum hybrid classifier built with Pennylane.

The model encodes a 4‑dimensional feature vector into a 4‑qubit
parametric circuit, runs a shallow ansatz, and reads out the expectation
value of Pauli‑Z on the first qubit.  The circuit is differentiable
via the parameter‑shift rule and is wrapped in a custom
torch.autograd.Function so that the entire network can be trained
with back‑propagation.  The quantum layer is followed by a sigmoid
to produce a probability, completing the binary classifier.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml


class QuantumCircuit:
    """A 4‑qubit variational circuit with a single layer of entanglement."""

    def __init__(self, n_qubits: int = 4, device: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> qml.QNode:
        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            # Encode each input feature as a Y‑rotation on its own qubit
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Entangle with a simple CZ pattern
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            # Ansatz layer
            for i in range(self.n_qubits):
                qml.RZ(params[i], wires=i)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def run(self, params: torch.Tensor) -> torch.Tensor:
        """Execute the circuit for the given parameters and return the expectation."""
        return self._circuit(params)


class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the Pennylane circuit using the
    parameter‑shift rule for gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        # Inputs are expected to be a 1‑D tensor of length n_qubits
        outputs = circuit.run(inputs)
        ctx.save_for_backward(inputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        # Compute gradients via parameter‑shift rule
        grads = []
        for i in range(inputs.shape[0]):
            shift_vec = torch.zeros_like(inputs)
            shift_vec[i] = shift
            forward = ctx.circuit.run(inputs + shift_vec)
            backward = ctx.circuit.run(inputs - shift_vec)
            grads.append((forward - backward) / (2 * shift))
        grad_inputs = torch.stack(grads).squeeze()
        return grad_inputs * grad_output, None, None


class HybridLayer(nn.Module):
    """Quantum layer that forwards a 4‑dimensional feature vector through the circuit."""

    def __init__(self, n_qubits: int = 4, device: str = "default.qubit", shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, device)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, n_qubits]
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            out = HybridFunction.apply(inputs[i], self.quantum_circuit, self.shift)
            outputs.append(out)
        return torch.stack(outputs).squeeze(-1)


class HybridClassifier(nn.Module):
    """Combines a simple linear preprocessing layer with the quantum hybrid layer."""

    def __init__(self, in_features: int = 4, n_qubits: int = 4, device: str = "default.qubit") -> None:
        super().__init__()
        self.preprocess = nn.Linear(in_features, n_qubits)
        self.quantum = HybridLayer(n_qubits, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, in_features]
        params = torch.tanh(self.preprocess(x))
        probs = torch.sigmoid(self.quantum(params))
        return torch.stack([probs, 1 - probs], dim=-1)


__all__ = ["HybridClassifier"]
