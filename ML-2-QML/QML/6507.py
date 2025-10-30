from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
from typing import Iterable

# ----- Photonic‑style variational circuit (Qiskit) -----

class QuantumPhotonicCircuit:
    def __init__(self, n_modes: int, backend, shots: int) -> None:
        self.n_modes = n_modes
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = qiskit.QuantumCircuit(n_modes)
        for i in range(n_modes):
            self._circuit.ry(self.theta, i)
        self._circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params],
        )
        job = self.backend.run(qobj)
        counts = job.result().get_counts()
        if isinstance(counts, list):
            return np.array([self._expectation(c) for c in counts])
        return np.array([self._expectation(counts)])

    def _expectation(self, counts: dict) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return np.sum(states * probs)

# ----- Autograd bridge -----

class PhotonicHybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumPhotonicCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.tolist()
        exp = circuit.run(np.array(angles))
        out = torch.tensor(exp, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for a in inputs.tolist():
            e_plus = ctx.circuit.run(np.array([a + shift]))
            e_minus = ctx.circuit.run(np.array([a - shift]))
            grad.append(e_plus - e_minus)
        grad = torch.tensor(grad, dtype=torch.float32)
        return grad * grad_output, None, None

# ----- Fraud‑detection quantum head only -----

class FraudDetectionHybrid(nn.Module):
    def __init__(self, n_modes: int = 2, shots: int = 100) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum = QuantumPhotonicCircuit(n_modes, backend, shots)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = PhotonicHybridFunction.apply(x, self.quantum, self.shift)
        probs = torch.sigmoid(probs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudDetectionHybrid"]
