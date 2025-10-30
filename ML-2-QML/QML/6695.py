"""Hybrid quantum‑classical binary classifier.

The quantum head is built on Qiskit’s EstimatorQNN, providing a
parameter‑shift differentiable expectation value that is used as the
output of the network.  The surrounding CNN mirrors the classical
counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer import AerSimulator

class QuantumEstimatorWrapper:
    """Wrapper around a one‑qubit EstimatorQNN circuit."""
    def __init__(self, backend, shots: int) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(1)
        self.input_param = Parameter("input1")
        self.weight_param = Parameter("weight1")
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        self.circuit.measure_all()
        self.observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = StatevectorEstimator()

    def run(self, input_value: float, weight_value: float) -> float:
        param_binds = [{self.input_param: input_value, self.weight_param: weight_value}]
        expectation = self.estimator.run(
            self.circuit,
            self.observable,
            param_binds,
            backend=self.backend,
            shots=self.shots
        )
        return float(expectation[0])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumEstimatorWrapper,
                shift: float, weight: torch.Tensor) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.weight = weight
        expectations = torch.tensor(
            [circuit.run(val.item(), weight.item()) for val in inputs],
            device=inputs.device,
            dtype=torch.float32
        )
        ctx.save_for_backward(inputs, expectations)
        return expectations

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, expectations = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        weight = ctx.weight

        # Gradient w.r.t inputs via parameter‑shift rule
        grad_inputs = []
        for val in inputs:
            right = circuit.run(val.item() + shift, weight.item())
            left = circuit.run(val.item() - shift, weight.item())
            grad_inputs.append((right - left) / (2 * shift))
        grad_inputs = torch.tensor(grad_inputs, device=grad_output.device, dtype=torch.float32)

        # Gradient w.r.t weight via parameter‑shift rule
        grad_weight_vals = []
        for val in inputs:
            right = circuit.run(val.item(), weight.item() + shift)
            left = circuit.run(val.item(), weight.item() - shift)
            grad_weight_vals.append((right - left) / (2 * shift))
        grad_weight = torch.sum(grad_output * torch.tensor(grad_weight_vals, device=grad_output.device, dtype=torch.float32))

        return grad_inputs * grad_output, None, None, grad_weight

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.circuit = QuantumEstimatorWrapper(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch,)
        return HybridFunction.apply(inputs, self.circuit, self.shift, self.weight)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        x = self.hybrid(x)
        return torch.stack((x, 1 - x), dim=-1)

__all__ = ["QuantumEstimatorWrapper", "HybridFunction", "Hybrid", "QCNet"]
