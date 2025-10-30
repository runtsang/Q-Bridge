"""Hybrid classical-quantum binary classifier with a quantum head.

This module implements the quantum counterpart of the QCNet architecture.
It uses a multi‑qubit variational ansatz derived from reference pair 2
and a parameter‑shift differentiable interface from reference pair 1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Build quantum classifier circuit from reference pair 2
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[qiskit.circuit.QuantumCircuit, list, list, list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = qiskit.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumCircuitWrapper:
    """Wrapper around a parametrised circuit executed on Aer."""
    def __init__(self, circuit: qiskit.circuit.QuantumCircuit, backend, shots: int) -> None:
        self._circuit = circuit
        self.backend = backend
        self.shots = shots
        self.params = list(circuit.parameters)
        self.n_qubits = circuit.num_qubits

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the parametrised circuit for the provided angles."""
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{param: val for param, val in zip(self.params, sample)}
                                         for sample in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        expectations = []
        for count_dict in result:
            exp = []
            for qubit in range(self.n_qubits):
                exp_val = sum(((-1)**int(bitstring[::-1][qubit])) * count
                              for bitstring, count in count_dict.items()) / self.shots
                exp.append(exp_val)
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Prepare full parameter vector: input features + zeros for weights
        n_qubits = circuit.n_qubits
        batch_size = inputs.shape[0]
        num_params = len(circuit.params)
        full_params = np.zeros((batch_size, num_params))
        full_params[:, :n_qubits] = inputs.cpu().numpy()
        expectation = ctx.circuit.run(full_params)
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        batch_size, n_qubits = inputs.shape
        gradients = []
        for idx in range(n_qubits):
            right = inputs.clone()
            right[:, idx] += shift
            expectation_right = ctx.circuit.run(right.cpu().numpy())
            left = inputs.clone()
            left[:, idx] -= shift
            expectation_left = ctx.circuit.run(left.cpu().numpy())
            grad_dim = np.sum(expectation_right - expectation_left, axis=1)
            gradients.append(grad_dim)
        gradients = np.stack(gradients, axis=1)  # shape (batch, n_qubits)
        gradients = torch.tensor(gradients, dtype=grad_output.dtype, device=grad_output.device)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        circuit, encoding, weights, observables = build_classifier_circuit(n_qubits, depth=1)
        self.quantum_circuit = QuantumCircuitWrapper(circuit, backend, shots)
        self.encoding = encoding
        self.weights = weights
        self.observables = observables
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, n_qubits)
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self, num_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_qubits)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(num_qubits, backend, shots=shots, shift=shift)
        self.logit_layer = nn.Linear(num_qubits, 1)

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
        x = self.fc3(x)  # shape (batch, num_qubits)
        x = self.hybrid(x)  # shape (batch, num_qubits)
        logit = self.logit_layer(x)  # shape (batch, 1)
        prob = torch.sigmoid(logit)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "QCNet", "build_classifier_circuit"]
