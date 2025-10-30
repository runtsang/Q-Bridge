"""Hybrid CNN + quantum expectation head.

The module builds a small two‑qubit variational circuit that
acts on the flattened feature vector from the convolutional
backbone.  The circuit is defined using Qiskit and evaluated
with the `StatevectorEstimator`.  A parameter‑shift rule is
used to provide analytical gradients back to PyTorch.
The design deliberately mirrors the classical module so that
the two can be swapped during ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit as QC
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np

class QuantumCircuitWrapper:
    """Two‑qubit variational circuit with a single input parameter."""
    def __init__(self) -> None:
        self.circuit = QC(2)
        self.theta = Parameter("theta")

        # Simple entangling block
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

        self.observable = SparsePauliOp.from_list([("Z", 1)])

    def run(self, params: list[float]) -> float:
        """Return expectation value of Z on qubit 1."""
        bound = self.circuit.assign_parameters({self.theta: params[0]}, inplace=False)
        state = Statevector.from_instruction(bound)
        return float(state.expectation_value(self.observable))

class QuantumHybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the parameterised quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp_val = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([exp_val], dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.view(-1).tolist():
            exp_plus  = ctx.circuit.run([val + shift])
            exp_minus = ctx.circuit.run([val - shift])
            grads.append(exp_plus - exp_minus)
        grad_tensor = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grad_tensor * grad_output, None, None

class QuantumHybrid(nn.Module):
    """Layer that forwards activations through the quantum circuit."""
    def __init__(self, shift: float = np.pi / 2, use_estimator_qnn: bool = False) -> None:
        super().__init__()
        self.shift = shift
        self.use_estimator_qnn = use_estimator_qnn
        if self.use_estimator_qnn:
            from qiskit.circuit import Parameter
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from qiskit_machine_learning.neural_networks import EstimatorQNN
            from qiskit.primitives import StatevectorEstimator

            qc = QuantumCircuit(1)
            theta_in = Parameter("theta")
            theta_w = Parameter("weight")
            qc.h(0)
            qc.ry(theta_in, 0)
            qc.rx(theta_w, 0)
            qc.measure_all()

            observable = SparsePauliOp.from_list([("Z", 1)])
            estimator = StatevectorEstimator()
            self.qnn = EstimatorQNN(
                circuit=qc,
                observables=observable,
                input_params=[theta_in],
                weight_params=[theta_w],
                estimator=estimator,
            )
        else:
            self.circuit = QuantumCircuitWrapper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_estimator_qnn:
            return self.qnn(x)
        else:
            return QuantumHybridFunction.apply(x, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN + quantum expectation head for binary classification."""
    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy = torch.zeros(1, 3, 32, 32, device=self.device)
        tmp = self._forward_conv(dummy)
        flat_features = tmp.shape[1]

        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.hybrid = QuantumHybrid(shift=np.pi / 2, use_estimator_qnn=False)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience evaluation on a batch of inputs."""
        self.eval()
        with torch.no_grad():
            return self(inputs)

__all__ = ["HybridBinaryClassifier"]
