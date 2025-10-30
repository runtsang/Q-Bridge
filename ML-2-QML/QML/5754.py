"""Quantum fusion head for the hybrid classifier.

A three‑qubit variational circuit is executed on the Aer simulator (or any
Qiskit backend).  The circuit returns the expectation value of the Z
operator on the first qubit, which is interpreted as a logit for binary
classification.  Gradients are computed with the parameter‑shift rule.
"""

import numpy as np
import torch
import torch.autograd as autograd
from typing import Optional
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumExpectationHead(torch.nn.Module):
    """Runs a parameterised 3‑qubit circuit and returns the expectation of Z."""
    def __init__(self,
                 backend: Optional[object] = None,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.shift = shift

    def _build_circuit(self, params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(3)
        for i, p in enumerate(params):
            circ.ry(p, i)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.measure_all()
        return circ

    def _expectation(self, circ: QuantumCircuit) -> float:
        compiled = transpile(circ, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, count in counts.items():
            # Z expectation: |0⟩ → +1, |1⟩ → -1 on the first qubit
            z = 1.0 if bitstring[-1] == '0' else -1.0
            exp += z * count
        return exp / self.shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a 1‑D tensor of parameters
        params = x.detach().cpu().numpy()
        circ = self._build_circuit(params)
        exp = self._expectation(circ)
        return torch.tensor([exp], dtype=torch.float32, device=x.device)

class HybridFunction(autograd.Function):
    """Differentiable wrapper implementing the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, head: QuantumExpectationHead, shift: float):
        ctx.head = head
        ctx.shift = shift
        exp = head.forward(inputs)
        return exp

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs = ctx.head
        shift = ctx.shift
        grads = []
        for i in range(inputs.shape[0]):
            pos = inputs.clone()
            neg = inputs.clone()
            pos[i] += shift
            neg[i] -= shift
            exp_pos = ctx.head.forward(pos)
            exp_neg = ctx.head.forward(neg)
            grads.append((exp_pos - exp_neg) / 2.0)
        grad_tensor = torch.stack(grads, dim=0)
        return grad_tensor * grad_output, None, None

class QuantumFusionHead(torch.nn.Module):
    """PyTorch module that produces a logit via a quantum circuit."""
    def __init__(self,
                 n_qubits: int = 3,
                 backend: Optional[object] = None,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        self.head = QuantumExpectationHead(backend, shots, shift)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.head, self.shift)

__all__ = ["QuantumExpectationHead",
           "HybridFunction",
           "QuantumFusionHead"]
