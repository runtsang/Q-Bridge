"""Hybrid quantum‑classical binary classifier.

The `HybridClassifier` class implements the same architecture as the
classical counterpart but replaces the final dense head with a
parameterised quantum circuit.  The circuit is built by the helper
`build_classifier_circuit` defined in this module and executed on the
Aer simulator.  The output is a single expectation value that is fed
through a sigmoid to obtain a probability for the positive class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int):
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
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


class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable wrapper that executes a quantum circuit and returns
    the expectation value of the first‑qubit Z observable."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: tuple, shift: float) -> torch.Tensor:
        # inputs shape: (batch, num_qubits)
        batch, num_qubits = inputs.shape
        expectations = []
        backend = Aer.get_backend("aer_simulator")
        for i in range(batch):
            param_bind = {param: float(inputs[i, j] + shift) for j, param in enumerate(circuit[1])}
            compiled = transpile(circuit[0], backend=backend)
            qobj = assemble(compiled, shots=1024, parameter_binds=[param_bind])
            job = backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, cnt in counts.items():
                # qubit 0 is the last bit in the bitstring
                z = 1 if bitstring[-1] == '0' else -1
                exp += z * cnt
            exp /= 1024
            expectations.append(exp)
        return torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # No analytic gradient; gradient will be approximated via the parameter shift rule
        # Here we return None to disable back‑prop through the quantum part
        return None, None, None


class HybridQuantum(nn.Module):
    """Quantum head that maps a feature vector to a single expectation value."""
    def __init__(self, num_qubits: int, depth: int, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = build_classifier_circuit(num_qubits, depth)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(x, self.circuit, self.shift)


class HybridClassifier(nn.Module):
    """End‑to‑end hybrid quantum‑classical binary classifier."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 1)
        self.drop = nn.Dropout2d(0.3)

        # Dense layers
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)  # latent dimension

        # Quantum head
        self.quantum = HybridQuantum(num_qubits=32, depth=2, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 32‑dim latent

        q_out = self.quantum(x)

        prob = torch.sigmoid(q_out)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["HybridClassifier"]
