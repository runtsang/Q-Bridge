import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.providers.aer import AerSimulator
from typing import Tuple

class QuantumExpectation:
    """Two‑qubit variational circuit that returns the expectation of Pauli‑Z."""
    def __init__(self, shots: int = 1000):
        self.backend = AerSimulator()
        self.shots = shots
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.barrier()
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the circuit for each input angle."""
        angles = inputs.squeeze().tolist()
        results = []
        for ang in angles:
            bound = {self.theta: ang}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bound])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            expectation = self._expectation(result)
            results.append(expectation)
        return torch.tensor(results, dtype=torch.float32).unsqueeze(0)

    def _expectation(self, counts: dict) -> float:
        total = sum(counts.values())
        exp = 0.0
        for state, cnt in counts.items():
            # Pauli‑Z expectation: (+1) for |00> or |11>, (-1) for |01> or |10>
            z = 1 if state in ("00", "11") else -1
            exp += z * cnt / total
        return exp

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper around the QuantumExpectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, expecter: QuantumExpectation, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.expecter = expecter
        out = expecter.run(inputs)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        inputs, out = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.squeeze().tolist():
            right = ctx.expecter.run(torch.tensor([val + shift]))
            left = ctx.expecter.run(torch.tensor([val - shift]))
            grads.append((right - left).item())
        grad = torch.tensor(grads, dtype=torch.float32).unsqueeze(0)
        return grad * grad_output, None, None

class HybridLayer(nn.Module):
    """Quantum expectation layer."""
    def __init__(self, expecter: QuantumExpectation, shift: float = 0.0):
        super().__init__()
        self.expecter = expecter
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.expecter, self.shift)

class FraudDetectionHybrid(nn.Module):
    """Quantum‑only variant that replaces the classical backbone with a variational circuit."""
    def __init__(self, shift: float = 0.0, shots: int = 1000):
        super().__init__()
        self.expecter = QuantumExpectation(shots=shots)
        self.hybrid = HybridLayer(self.expecter, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        prob = self.hybrid(inputs)
        prob = torch.sigmoid(prob)
        probs = torch.cat([prob, 1 - prob], dim=-1)
        return probs

__all__ = ["QuantumExpectation", "HybridLayer", "FraudDetectionHybrid"]
