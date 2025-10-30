"""Quantum‑enabled estimator mirroring the classical EstimatorQNN.

The module builds a two‑qubit variational circuit, evaluates the
expectation value of a Pauli operator, and exposes the same
EstimatorQNN interface that the classical module provides.  The
quantum head is parameterised by the output of the classical backbone
and feeds into an expectation operator that is differentiable
through the parameter‑shift rule.
"""

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer, execute, transpile, assemble
from qiskit.circuit import Parameter
from typing import Callable, Optional

class QuantumCircuitWrapper:
    """
    Two‑qubit variational circuit returning the expectation of the Pauli‑Y operator
    on the second qubit.  Uses the state‑vector simulator for exact values.
    """
    def __init__(self, backend=Aer.get_backend("statevector_simulator")) -> None:
        self.backend = backend
        self.theta = Parameter("theta")

        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.cx(0, 1)

    def expectation(self, params: np.ndarray) -> np.ndarray:
        """Return expectation values for an array of parameters."""
        exp_vals = []
        for p in params:
            bound = self.circuit.bind_parameters({self.theta: p})
            result = execute(bound, self.backend).result()
            sv = result.get_statevector()
            # Pauli‑Y expectation on qubit 1
            pauli = qiskit.quantum_info.Pauli("Y")
            exp = np.real(pauli.expectation_value(sv))
            exp_vals.append(exp)
        return np.array(exp_vals)

class QuantumExpectationFunction(torch.autograd.Function):
    """
    Custom autograd function for the parameter‑shift rule.  Forward
    evaluates the quantum expectation; backward applies the finite‑difference
    gradient.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum: QuantumCircuitWrapper,
                shift: float) -> torch.Tensor:
        ctx.quantum = quantum
        ctx.shift = shift
        exp = quantum.expectation(inputs.detach().cpu().numpy())
        ctx.save_for_backward(inputs)
        return torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        quantum = ctx.quantum
        grad_inputs = []
        for val in inputs:
            plus = quantum.expectation(np.array([val.item() + shift]))
            minus = quantum.expectation(np.array([val.item() - shift]))
            grad = (plus - minus) / (2 * np.sin(shift))
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

class HybridLayerQuantum(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, quantum: QuantumCircuitWrapper, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum = quantum
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumExpectationFunction.apply(x, self.quantum, self.shift)

class EstimatorQNN(nn.Module):
    """
    Same signature as the classical EstimatorQNN but with a quantum head.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_features: int = 8,
                 n_outputs: int = 1,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.Tanh(),
            nn.Linear(hidden_features // 2, hidden_features // 4),
            nn.Tanh(),
        )
        self.quantum_circuit = QuantumCircuitWrapper()
        self.shift = shift
        if n_outputs == 2:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        quantum_out = HybridLayerQuantum(self.quantum_circuit, self.shift)(x)
        if hasattr(self, "sigmoid"):
            prob = self.sigmoid(quantum_out)
            return torch.cat([prob, 1 - prob], dim=-1)
        return quantum_out

__all__ = ["EstimatorQNN"]
