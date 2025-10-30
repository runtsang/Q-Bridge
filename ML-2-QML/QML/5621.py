"""Quantum‑classical estimator that mirrors the classical counterpart.

The module implements:
* Variational circuit construction (mirrors the quantum `build_classifier_circuit`).
* A lightweight wrapper around a Qiskit circuit for state‑vector simulation.
* A `HybridFunction` that forwards activations through the quantum circuit and supplies gradients via finite differences.
* `HybridEstimatorQNN` – a PyTorch module that exposes the same interface as the classical implementation.

This design keeps the QML side fully quantum‑centric while preserving a compatible API."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from typing import Iterable, Sequence, Tuple, List

# --------------------------------------------------------------------------- #
# Quantum classifier construction (mirrors QuantumClassifierModel.py)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[qiskit.circuit.QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Create a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = qiskit.circuit.QuantumCircuit(num_qubits)
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

# --------------------------------------------------------------------------- #
# Wrapper for executing a parameterised circuit on the Aer simulator
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """Executes a parameterised circuit on the Aer simulator."""
    def __init__(self, circuit: qiskit.circuit.QuantumCircuit, shots: int = 1024):
        self.circuit = circuit
        self.backend = Aer.get_backend("aer_simulator")
        self.shots = shots

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: val} for p, val in zip(self.circuit.parameters, params)],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# Hybrid function that forwards activations through the quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        out = torch.tensor(expectation)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for val, s in zip(inputs.tolist(), shift):
            grads.append(ctx.circuit.run([val + s])[0] - ctx.circuit.run([val - s])[0])
        return torch.tensor(grads).float() * grad_output, None, None

# --------------------------------------------------------------------------- #
# Main estimator module
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """Quantum‑classical estimator that mirrors the classical counterpart."""
    def __init__(self, num_qubits: int, depth: int, shift: float = np.pi / 2, shots: int = 1024, device: str = "cpu") -> None:
        super().__init__()
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.quantum_circuit = QuantumCircuitWrapper(self.circuit, shots)
        self.shift = shift
        self.device = device
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = HybridFunction.apply(inputs, self.quantum_circuit, self.shift)
        return torch.stack((probs, 1 - probs), dim=-1)
