from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, transpile
from qiskit.compiler import assemble
from qiskit.circuit import Parameter

class QuantumCircuit:
    """
    A simple two‑qubit variational circuit used as the quantum head.
    The circuit consists of an H‑gate on each qubit, a chain of CX
    entanglements, followed by a parameterised RX rotation on every
    qubit.  The expectation value of the Z‑observable on the first
    qubit is returned.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self.circuit.h(q)
        for q in range(n_qubits - 1):
            self.circuit.cx(q, q + 1)
        self.circuit.rx(self.theta, list(range(n_qubits)))
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter values.
        Returns the expectation of Z on the first qubit for each
        parameter in the batch.
        """
        compiled = transpile(self.circuit, self.backend)
        exp_vals = []
        for theta in thetas:
            param_bind = [{self.theta: theta}]
            qobj = assemble(compiled, shots=self.shots, parameter_binds=param_bind)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                exp += (1 if state[-1] == "0" else -1) * cnt
            exp /= self.shots
            exp_vals.append(exp)
        return np.array(exp_vals, dtype=np.float32)

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit.
    The forward pass evaluates the expectation value; the backward
    pass uses a centred finite‑difference to estimate the gradient.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectations = circuit.run(thetas)
        return torch.tensor(expectations, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        grad = []
        for val in grad_output.detach().cpu().numpy():
            exp_plus = ctx.circuit.run(np.array([val + shift]))
            exp_minus = ctx.circuit.run(np.array([val - shift]))
            grad.append((exp_plus - exp_minus) / (2 * shift))
        grad = torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """
    Quantum‑conditioned dense layer.  It forwards a scalar tensor
    through the variational circuit and returns the corresponding
    expectation value.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(-1), self.circuit, self.shift)

class HybridBinaryRegressionNet(nn.Module):
    """
    Hybrid network that merges the classical CNN backbone with the
    quantum‑conditioned dense head.  The network can be used for
    binary classification or regression by toggling the ``classification``
    flag.  The architecture mirrors the classical version but replaces
    the final linear head with a quantum layer.
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_filters: int = 32,
        dropout: float = 0.3,
        n_qubits: int = 2,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
        classification: bool = True,
    ) -> None:
        super().__init__()
        self.classification = classification
        # Residual block 1
        self.conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv1_res = nn.Conv2d(input_channels, base_filters, kernel_size=1, stride=1, padding=0)
        # Residual block 2
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.conv2_res = nn.Conv2d(base_filters, base_filters * 2, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        # Dense head
        self.fc1 = nn.Linear((base_filters * 2) * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, classification: bool | None = None) -> torch.Tensor:
        if classification is None:
            classification = self.classification
        # Residual block 1
        res = self.conv1_res(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x + res
        x = self.drop(x)
        # Residual block 2
        res = self.conv2_res(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x + res
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.hybrid(x)
        if classification:
            probs = self.sigmoid(x)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            return x.squeeze(-1)

__all__ = ["HybridBinaryRegressionNet"]
