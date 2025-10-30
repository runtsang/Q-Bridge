"""HybridQCNet: Quantum‑enhanced binary classifier.

This module builds upon the classical HybridQCNet and replaces its dense head
with a differentiable quantum expectation layer.  The quantum head is a
parameterised 2‑qubit circuit executed on the Aer simulator.  A lightweight
fully‑connected quantum sub‑layer is also exposed for experimentation.
"""

from __future__ import annotations

import importlib
import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import transpile, assemble

# Import the classical backbone
_ml_module = importlib.import_module("HybridQCNet")
ClassicalHybridQCNet = _ml_module.HybridQCNet

class QuantumCircuit:
    """Two‑qubit parameterised circuit executed on a classical backend."""

    def __init__(self, n_qubits: int, backend, shots: int):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        job = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        result = self.backend.run(job).result().get_counts()
        if isinstance(result, list):
            return np.array([self._expectation(c) for c in result])
        return np.array([self._expectation(result)])

    def _expectation(self, counts: dict) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys())).astype(float)
        return float(np.sum(states * probs))

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards a scalar to a quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            e_plus = ctx.circuit.run([val + shift])[0]
            e_minus = ctx.circuit.run([val - shift])[0]
            grads.append(e_plus - e_minus)
        grad = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None

class QuantumHybridLayer(nn.Module):
    """Layer that maps a real scalar to a quantum expectation value."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class FullyConnectedQuantumLayer(nn.Module):
    """A lightweight fully‑connected quantum sub‑layer.

    The circuit implements a single‑qubit Ry rotation per input feature and
    returns the expectation of Pauli‑Z, effectively providing a learnable
    linear mapping in the quantum domain.
    """

    def __init__(self, n_qubits: int, backend, shots: int):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thetas = x.detach().cpu().numpy().flatten()
        expectations = self.circuit.run(thetas)
        return torch.tensor(expectations, device=x.device, dtype=x.dtype).view(x.shape[0], -1)

class HybridQCNet(ClassicalHybridQCNet):
    """CNN + quantum expectation head for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = QuantumHybridLayer(
            n_qubits=self.fc3.out_features,
            backend=backend,
            shots=200,
            shift=np.pi / 2,
        )
        # Optional fully‑connected quantum sub‑layer
        self.fully_connected_q = FullyConnectedQuantumLayer(
            n_qubits=self.fc3.out_features,
            backend=backend,
            shots=200,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical forward to get the probability of the positive class
        probs = super().forward(x)[:, 0].unsqueeze(-1)
        # Pass the probability through the quantum head
        q_out = self.quantum_head(probs)
        return torch.cat([q_out, 1 - q_out], dim=-1)

__all__ = ["HybridQCNet"]
