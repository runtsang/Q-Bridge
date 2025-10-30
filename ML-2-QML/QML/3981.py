"""Quantum utilities for HybridQuantumClassifier.

Defines:
- QuantumHybridLayer: a variational layer that maps a scalar to a probability.
- QuantumHybridFunction: autograd function that evaluates the circuit and its gradient.
- QuantumLSTMCell: LSTM cell where each gate is realised by a small quantum circuit.
- HybridQuantumClassifier: a pure‑quantum classifier that can serve as a drop‑in replacement.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import Aer, assemble, transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.aer import AerSimulator

class QuantumHybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, params: torch.Tensor, backend, shots: int):
        ctx.backend = backend
        ctx.shots = shots
        ctx.save_for_backward(x, params)
        batch, _ = x.shape
        results = []
        for i in range(batch):
            angle = x[i, 0].item()
            qc = QuantumCircuit(2)
            qc.h(range(2))
            qc.ry(angle, 0)
            qc.measure_all()
            compiled = transpile(qc, backend)
            qobj = assemble(compiled, shots=shots)
            job = backend.run(qobj)
            counts = job.result().get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                z = 1 if state[0] == "0" else -1
                exp += z * cnt
            exp /= shots
            results.append(exp)
        probs = torch.tensor(results, dtype=x.dtype, device=x.device)
        return torch.sigmoid(probs)

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors
        shift = np.pi / 2
        grad_x = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                pos = x[i, j].item() + shift
                neg = x[i, j].item() - shift
                # Positive evaluation
                qc_pos = QuantumCircuit(2)
                qc_pos.h(range(2))
                qc_pos.ry(pos, 0)
                qc_pos.measure_all()
                compiled_pos = transpile(qc_pos, ctx.backend)
                job_pos = ctx.backend.run(assemble(compiled_pos, shots=ctx.shots))
                exp_pos = 0.0
                for state, cnt in job_pos.result().get_counts().items():
                    z = 1 if state[0] == "0" else -1
                    exp_pos += z * cnt
                exp_pos /= ctx.shots
                # Negative evaluation
                qc_neg = QuantumCircuit(2)
                qc_neg.h(range(2))
                qc_neg.ry(neg, 0)
                qc_neg.measure_all()
                compiled_neg = transpile(qc_neg, ctx.backend)
                job_neg = ctx.backend.run(assemble(compiled_neg, shots=ctx.shots))
                exp_neg = 0.0
                for state, cnt in job_neg.result().get_counts().items():
                    z = 1 if state[0] == "0" else -1
                    exp_neg += z * cnt
                exp_neg /= ctx.shots
                grad = (exp_pos - exp_neg) / (2 * shift)
                grad_x[i, j] = grad * grad_output[i]
        return grad_x, None, None, None

class QuantumHybridLayer(nn.Module):
    """Variational layer that maps a scalar to a probability using a 2‑qubit circuit."""
    def __init__(self, in_features: int, n_qubits: int = 2, shots: int = 1024, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.params = nn.Parameter(torch.randn(in_features))

    def forward(self, x: torch.Tensor):
        return QuantumHybridFunction.apply(x, self.params, self.backend, self.shots)

class QuantumLSTMCell(nn.Module):
    """Quantum‑enhanced LSTM cell with 2‑qubit circuits for each gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 2, shots: int = 1024, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()

        self.forget_lin = nn.Linear(input_dim + hidden_dim, 1)
        self.input_lin = nn.Linear(input_dim + hidden_dim, 1)
        self.update_lin = nn.Linear(input_dim + hidden_dim, 1)
        self.output_lin = nn.Linear(input_dim + hidden_dim, 1)

    def _quantum_gate(self, angle: torch.Tensor):
        batch = angle.shape[0]
        results = []
        for i in range(batch):
            a = angle[i, 0].item()
            qc = QuantumCircuit(self.n_qubits)
            qc.h(range(self.n_qubits))
            qc.ry(a, 0)
            qc.measure_all()
            compiled = transpile(qc, self.backend)
            job = self.backend.run(assemble(compiled, shots=self.shots))
            counts = job.result().get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                z = 1 if state[0] == "0" else -1
                exp += z * cnt
            exp /= self.shots
            results.append(exp)
        return torch.tensor(results, dtype=angle.dtype, device=angle.device)

    def forward(self, x: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(x, states)
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_lin(combined))
        i = torch.sigmoid(self.input_lin(combined))
        g = torch.tanh(self.update_lin(combined))
        o = torch.sigmoid(self.output_lin(combined))

        f = self._quantum_gate(f)
        i = self._quantum_gate(i)
        g = self._quantum_gate(g)
        o = self._quantum_gate(o)

        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, (hx, cx)

    def _init_states(self, x: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = x.size(0)
        device = x.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class HybridQuantumClassifier(nn.Module):
    """Pure‑quantum classifier that can replace the image mode of the hybrid model."""
    def __init__(self, input_dim: int, n_qubits: int = 2, shots: int = 1024):
        super().__init__()
        self.quantum_layer = QuantumHybridLayer(input_dim, n_qubits=n_qubits, shots=shots)

    def forward(self, x: torch.Tensor):
        return self.quantum_layer(x)

__all__ = ["QuantumHybridLayer", "QuantumHybridFunction", "QuantumLSTMCell", "HybridQuantumClassifier"]
