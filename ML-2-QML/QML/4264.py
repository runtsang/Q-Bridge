"""Hybrid quantum‑classical binary classifier.

The model mirrors the classical backbone and replaces the final
dense head with a parameter‑shifted quantum expectation layer.
The quantum circuit is a single‑qubit Ry–Rx circuit whose parameter
is produced by a tiny regression head on the classical features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.transpile import transpile
from qiskit.compiler import assemble

class QuantumCircuitWrapper:
    """Simple 1‑qubit circuit with a single trainable rotation."""
    def __init__(self, backend: any, shots: int = 100) -> None:
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.rx(self.theta, 0)
        self.circuit.measure_all()
        self.transpiled = transpile(self.circuit, backend)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Return expectation of Z for each theta in thetas."""
        results = []
        for theta in np.atleast_1d(thetas):
            bound = self.circuit.bind_parameters({self.theta: theta})
            qobj = assemble(
                self.transpiled,
                shots=self.shots,
                parameter_binds=[{self.theta: theta}],
            )
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            # Convert counts to expectation of Pauli‑Z
            exp = 0.0
            for bitstring, count in counts.items():
                prob = count / self.shots
                bit = int(bitstring, 2)
                exp += (bit * 2 - 1) * prob  # Z eigenvalue: -1 for |0>, +1 for |1>
            results.append(exp)
        return np.array(results)

class QuantumFunction(torch.autograd.Function):
    """Differentiable wrapper that evaluates the quantum expectation
    and implements the parameter‑shift rule for gradients."""
    @staticmethod
    def forward(ctx, theta: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.run(theta.detach().cpu().numpy())
        result = torch.tensor(exp, device=theta.device, dtype=theta.dtype)
        ctx.save_for_backward(theta, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        theta, _ = ctx.saved_tensors
        shift = ctx.shift
        exp_plus = ctx.circuit.run((theta + shift).detach().cpu().numpy())
        exp_minus = ctx.circuit.run((theta - shift).detach().cpu().numpy())
        grad_theta = (exp_plus - exp_minus) / (2 * np.sin(shift))
        return grad_theta * grad_output, None, None

class HybridBinaryClassifier(nn.Module):
    """Quantum‑classical hybrid binary classifier."""
    def __init__(self) -> None:
        super().__init__()
        # Classical backbone identical to the ML version
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Regression head that produces the rotation angle for the quantum circuit
        self.shift_head = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        # Quantum head
        self.quantum = QuantumCircuitWrapper(Aer.get_backend("aer_simulator"))
        self.shift = np.pi / 2  # parameter‑shift step

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a probability vector of shape (B,2)."""
        feats = self.features(x)
        feats = self.norm(feats)
        theta = self.shift_head(feats).squeeze(-1)  # (B,)
        # Evaluate quantum expectation with parameter‑shift gradients
        exp = QuantumFunction.apply(theta, self.quantum, self.shift)
        probs = self.sigmoid(exp)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
