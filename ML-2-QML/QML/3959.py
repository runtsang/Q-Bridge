"""Hybrid classical-quantum binary classification: quantum variant.

This module defines a PyTorch model that mirrors the classical counterpart but
uses a parameterised quantum circuit as the prediction head.  It leverages
Qiskit’s Aer simulator and offers a modular factory for building the circuit
and associated metadata.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, assemble, transpile, Aer
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list, list, list[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, list of encoding parameters, list of weight parameters,
    and a list of observables for expectation evaluation.
    """
    encoding = [f"x{i}" for i in range(num_qubits)]
    weights = [f"theta{j}" for j in range(num_qubits * depth)]

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)
    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    # Observables: single-qubit Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

class QuantumCircuitWrapper:
    """
    Wrapper around a parametrised circuit executed on Aer.
    """
    def __init__(self, circuit: QuantumCircuit, backend, shots: int) -> None:
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.params = list(circuit.parameters)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{p: t} for p, t in zip(self.params, thetas)])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        else:
            return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable interface between PyTorch and the quantum circuit.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        gradients = []
        for idx, val in enumerate(inputs.numpy()):
            right = ctx.circuit.run([val + shift[idx]])[0]
            left = ctx.circuit.run([val - shift[idx]])[0]
            gradients.append(right - left)
        grad = torch.tensor(gradients, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, num_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        circuit, _, _, _ = build_classifier_circuit(num_qubits, depth=2)
        self.quantum_circuit = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.view(-1)
        return HybridFunction.apply(flat, self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.
    """
    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(1, backend, shots=100, shift=shift)

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
        p = self.hybrid(x).squeeze(-1)
        probs = torch.stack([p, 1 - p], dim=-1)
        return probs

__all__ = ["build_classifier_circuit", "QuantumCircuitWrapper", "HybridFunction", "Hybrid", "QCNet"]
