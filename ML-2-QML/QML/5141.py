"""Quantum module providing a hybrid quantum expectation head and self‑attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parametrised single‑qubit circuit for expectation evaluation."""
    def __init__(self, backend: qiskit.providers.Backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
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
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            return np.sum(states * probs)
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum expectation circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        gradients = []
        for theta in thetas:
            right = ctx.circuit.run([theta + shift])[0]
            left = ctx.circuit.run([theta - shift])[0]
            gradients.append(right - left)
        grad_inputs = torch.tensor(gradients, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")
        self.backend = AerSimulator()

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Execute the attention circuit and return expectation values for each qubit."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        result = job.result().get_counts(circuit)
        expectations = np.zeros(self.n_qubits)
        for state, count in result.items():
            prob = count / shots
            bits = np.array([int(b) for b in state[::-1]])  # little‑endian
            expectations += bits * prob
        return expectations

class HybridBinaryClassifier(nn.Module):
    """Quantum expectation head that can be used as the final layer of a hybrid model."""
    def __init__(self, shift: float = np.pi / 2, shots: int = 200):
        super().__init__()
        backend = AerSimulator()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, 1)
        return HybridFunction.apply(inputs.squeeze(-1), self.circuit, self.shift).unsqueeze(-1)
