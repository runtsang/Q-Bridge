"""Quantum‑enhanced binary classifier built on a classical backbone.

The class shares the public API of the classical version but injects
a small variational circuit that produces a scalar expectation value.
Differentiation is achieved via a custom autograd function that
implements a finite‑difference gradient.
"""

import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

class _QuantumCircuit:
    """
    Lightweight wrapper around a parametrised two‑qubit circuit
    executed on Aer.  The circuit consists of H gates, an
    RX(θ) rotation, and measurement of all qubits.  The
    expectation value of the computational basis states is
    returned.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.backend = AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter('θ')
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def expectation(self, theta: float) -> float:
        bound = self.circuit.bind_parameters({self.theta: theta})
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        return float(np.sum(states * probs))

class _QuantumHead(torch.autograd.Function):
    """
    Autograd wrapper that forwards the classical logits through
    the quantum circuit and returns a scalar expectation value.
    The backward pass uses a central finite‑difference approximation.
    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor, circuit: _QuantumCircuit, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        theta = logits.detach().cpu().numpy()
        exp_vals = np.array([circuit.expectation(t) for t in theta])
        out = torch.from_numpy(exp_vals).to(logits.device)
        ctx.save_for_backward(logits)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in logits.detach().cpu().numpy():
            g = (ctx.circuit.expectation(val + shift) -
                 ctx.circuit.expectation(val - shift)) / (2 * shift)
            grads.append(g)
        grad_inputs = torch.tensor(grads, device=logits.device) * grad_output
        return grad_inputs, None, None

class QuantumHybridClassifier(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.
    The quantum head is a two‑qubit circuit whose parameter is the
    output of the final linear layer.
    """
    def __init__(self, in_features: int, n_qubits: int = 2,
                 shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.circuit = _QuantumCircuit(n_qubits, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        q_score = _QuantumHead.apply(logits.squeeze(-1), self.circuit, self.shift)
        logits = logits + q_score
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
