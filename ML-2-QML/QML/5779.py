"""Quantum‑classical hybrid binary classifier.

The quantum head is constructed by `build_classifier_circuit` and
executed on a state‑vector simulator.  A differentiable `HybridFunction`
wraps the expectation value, enabling gradient flow through the circuit.
The preceding convolutional backbone is identical to the classical
implementation, ensuring a fair comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Mirrors the classical `build_classifier_circuit` in structure and metadata.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumCircuitExecutor:
    """Executes a parametrised circuit on Aer and returns expectation values."""
    def __init__(self, circuit: QuantumCircuit, backend: AerSimulator, shots: int = 1024) -> None:
        self.circuit = circuit
        self.backend = backend
        self.shots = shots

    def run(self, params: list[float]) -> float:
        """Evaluate the circuit for a single set of parameters."""
        compiled = transpile(self.circuit, self.backend)
        param_bind = {self.circuit.parameters[i]: params[i] for i in range(len(params))}
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_bind])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)

        return expectation(result)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge to the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, executor: QuantumCircuitExecutor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.executor = executor
        # Convert to numpy for circuit evaluation
        exp_val = executor.run(inputs.detach().cpu().numpy().tolist())
        out = torch.tensor([exp_val], device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        executor = ctx.executor
        # Finite‑difference gradient
        grad = executor.run([inputs.item() + shift]) - executor.run([inputs.item() - shift])
        return torch.tensor([grad], device=grad_output.device, dtype=grad_output.dtype) * grad_output, None, None

class HybridQuantumLayer(nn.Module):
    """Layer that forwards activations through a variational circuit."""
    def __init__(self, num_qubits: int, depth: int, shots: int = 1024, shift: float = np.pi/2) -> None:
        super().__init__()
        circuit, _, _, _ = build_classifier_circuit(num_qubits, depth)
        self.executor = QuantumCircuitExecutor(circuit, AerSimulator(), shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.executor, self.shift)

class QCNetQuantum(nn.Module):
    """Convolutional backbone followed by a quantum hybrid head."""
    def __init__(self, num_qubits: int = 1, depth: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # output size matches qubit count
        self.hybrid = HybridQuantumLayer(num_qubits, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        x = self.hybrid(x).squeeze(-1)
        probs = torch.sigmoid(x)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["build_classifier_circuit", "QuantumCircuitExecutor", "HybridQuantumLayer", "QCNetQuantum"]
