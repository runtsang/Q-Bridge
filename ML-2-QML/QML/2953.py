"""Quantum hybrid binary classifier with CNN backbone and EstimatorQNN head.

The network shares the same interface as the classical version but replaces the
final sigmoid head with a parameter‑shift differentiable quantum expectation
calculated by a one‑qubit circuit.  The quantum circuit is wrapped in a
torch.autograd.Function for seamless gradient propagation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli
from qiskit.circuit import Parameter

class QuantumExpectation(torch.autograd.Function):
    """
    Differentiable interface to a 1‑qubit expectation value.
    Uses parameter‑shift rule for both input and weight parameters.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor,
                backend, compiled_circuit: QuantumCircuit,
                weight: torch.Tensor,
                shift: float) -> torch.Tensor:
        ctx.backend = backend
        ctx.compiled_circuit = compiled_circuit
        ctx.weight = weight
        ctx.shift = shift

        # Compute expectation for each input value
        expectations = []
        for inp in inputs.detach().cpu().numpy():
            bound = compiled_circuit.bind_parameters({
                compiled_circuit.parameters[0]: inp,
                compiled_circuit.parameters[1]: weight.item()
            })
            state = Statevector(bound)
            Y = Pauli('Y').to_matrix()
            exp = np.real(state.expectation_value(Y))
            expectations.append(exp)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, weight)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, weight = ctx.saved_tensors
        shift = ctx.shift
        compiled_circuit = ctx.compiled_circuit

        # Gradient w.r.t. inputs via parameter‑shift
        grad_inputs = []
        for inp in inputs.detach().cpu().numpy():
            plus = compiled_circuit.bind_parameters({
                compiled_circuit.parameters[0]: inp + shift,
                compiled_circuit.parameters[1]: weight.item()
            })
            minus = compiled_circuit.bind_parameters({
                compiled_circuit.parameters[0]: inp - shift,
                compiled_circuit.parameters[1]: weight.item()
            })
            state_plus = Statevector(plus)
            state_minus = Statevector(minus)
            Y = Pauli('Y').to_matrix()
            exp_plus = np.real(state_plus.expectation_value(Y))
            exp_minus = np.real(state_minus.expectation_value(Y))
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)

        # Gradient w.r.t. weight via parameter‑shift
        grad_weight = []
        for inp in inputs.detach().cpu().numpy():
            plus = compiled_circuit.bind_parameters({
                compiled_circuit.parameters[0]: inp,
                compiled_circuit.parameters[1]: weight.item() + shift
            })
            minus = compiled_circuit.bind_parameters({
                compiled_circuit.parameters[0]: inp,
                compiled_circuit.parameters[1]: weight.item() - shift
            })
            state_plus = Statevector(plus)
            state_minus = Statevector(minus)
            Y = Pauli('Y').to_matrix()
            exp_plus = np.real(state_plus.expectation_value(Y))
            exp_minus = np.real(state_minus.expectation_value(Y))
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_weight.append(grad)
        # Weight gradient is average over batch
        grad_weight = np.mean(grad_weight)
        grad_weight = torch.tensor(grad_weight, dtype=weight.dtype, device=weight.device)
        grad_weight = grad_weight * grad_output.sum()

        # Multiply with upstream gradient
        return grad_inputs * grad_output, None, None, grad_weight, None

class HybridBinaryClassifier(nn.Module):
    """
    Quantum hybrid classifier mirroring the classical backbone.
    Replaces the sigmoid head with a differentiable quantum expectation.
    """
    def __init__(self,
                 backend = Aer.get_backend("aer_simulator"),
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum circuit (1‑qubit, input + weight)
        qc = QuantumCircuit(1)
        input_param = Parameter("input")
        weight_param = Parameter("weight")
        qc.h(0)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)
        qc.measure_all()
        compiled = transpile(qc, backend)
        self.compiled_circuit = compiled

        # Trainable weight parameter for the quantum circuit
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.shift = shift
        self.backend = backend
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Fully‑connected reduction
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # shape (batch, 1)

        # Quantum expectation head
        probs = QuantumExpectation.apply(
            x.squeeze(-1),
            self.backend,
            self.compiled_circuit,
            self.weight,
            self.shift
        )
        probs = self.sigmoid(probs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
