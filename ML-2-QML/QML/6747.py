"""Hybrid binary classifier with a quantum fully connected head.

This module implements the same convolutional feature extractor as the
classical counterpart but replaces the final fully connected layer with a
parameterised quantum circuit.  The circuit is executed on a Qiskit Aer
simulator and the expectation value is back‑propagated using a custom
autograd function.
"""

import qiskit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import assemble, transpile


class QuantumFullyConnectedCircuit:
    """Parametrised two‑qubit circuit that returns the expectation of the
    Z‑observable.  The circuit is reused for every forward pass and
    executed on the provided backend.
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
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class QuantumHybridFunction(torch.autograd.Function):
    """Custom autograd function that forwards the input to a quantum circuit
    and returns its expectation value.  The backward pass approximates the
    gradient using a finite‑difference scheme with a shift.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumFullyConnectedCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grad_inputs = []
        for val, s in zip(inputs.tolist(), shift):
            right = ctx.circuit.run([val + s])
            left = ctx.circuit.run([val - s])
            grad_inputs.append(right - left)
        grad = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad * grad_output, None, None


class QuantumHybridHead(nn.Module):
    """Hybrid layer that transforms classical activations through a quantum
    circuit.  It expects a 1‑D input tensor of shape (batch,).
    """
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumFullyConnectedCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)


class HybridBinaryClassifier(nn.Module):
    """Convolutional binary classifier that ends with a quantum fully connected
    layer.  The architecture mirrors the classical reference and can be
    trained end‑to‑end using standard PyTorch optimisers.
    """
    def __init__(self, use_quantum: bool = True) -> None:
        super().__init__()
        if not use_quantum:
            raise ValueError("Classical head is not available in the quantum module.")
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.head = QuantumHybridHead(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

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
        logits = self.head(x)
        prob = torch.sigmoid(logits)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["QuantumFullyConnectedCircuit", "QuantumHybridFunction",
           "QuantumHybridHead", "HybridBinaryClassifier"]
