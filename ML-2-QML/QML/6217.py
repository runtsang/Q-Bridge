"""Hybrid CNN + quantum expectation classifier.

The model extracts features with a CNN backbone and feeds them to a parameterised quantum circuit
whose expectation value is used as the logit for binary classification. The circuit is executed
on the Aer simulator with a fixed number of shots. The architecture is fully differentiable
via a custom autograd function.

The design demonstrates how a classical neural network can be coupled with a quantum
sub‑network to potentially capture richer feature interactions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer
from qiskit import assemble, transpile


class QuantumCircuit:
    """Two‑qubit parametrised circuit that returns the expectation of Z on the first qubit."""
    def __init__(self, backend, shots: int = 100) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that evaluates the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.tolist()):
            exp_right = ctx.quantum_circuit.run([value + shift[idx]])
            exp_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(exp_right - exp_left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None


class Hybrid(nn.Module):
    """Quantum head that forwards activations through the circuit."""
    def __init__(self, in_features: int, backend, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(backend, shots)
        self.linear = nn.Linear(in_features, 1)  # map features to a single angle
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        angles = self.linear(inputs).squeeze(-1)
        return HybridFunction.apply(angles, self.quantum_circuit, self.shift)


class HybridKernelClassifier(nn.Module):
    """CNN backbone followed by a quantum expectation head and linear output."""
    def __init__(self, in_channels: int = 3, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.2),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, in_channels, 32, 32)
        feat_dim = self.backbone(dummy).shape[1]
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(feat_dim, backend, shots, shift)
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        quantum_output = self.hybrid(features).squeeze(-1)
        logits = self.classifier(quantum_output.unsqueeze(-1)).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.cat([probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)], dim=-1)


__all__ = ["HybridKernelClassifier", "Hybrid", "HybridFunction", "QuantumCircuit"]
