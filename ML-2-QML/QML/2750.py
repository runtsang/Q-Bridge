"""Hybrid binary classifier with a pure quantum head.

The module exposes a single class `HybridQuantumBinaryClassifier` that
always uses a variational two‑qubit circuit as the final layer.  The
CNN backbone is identical to the classical version, but the head
is implemented with a differentiable expectation value computed on
Qiskit Aer.  The design mirrors the classical module but removes
the optional dense branch, making the quantum contribution
central to the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile

class _TwoQubitVariationalCircuit:
    """Variational circuit with two qubits executed on Aer."""
    def __init__(self, shots: int = 100):
        self.backend = qiskit.Aer.get_backend('aer_simulator')
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter('theta')
        # Entangling pattern
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()
        self.shots = shots

    def run(self, theta: float):
        bound = self.circuit.bind_parameters({self.theta: theta})
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = {k: v / self.shots for k, v in counts.items()}
        # expectation of Z on first qubit
        exp = sum(((-1)**int(bit[0])) * p for bit, p in probs.items())
        return exp

class _QuantumExpectationLayer(nn.Module):
    """Differentiable wrapper that forwards a scalar through the circuit."""
    class _Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta: torch.Tensor, circuit: _TwoQubitVariationalCircuit, shift: float):
            ctx.circuit = circuit
            ctx.shift = shift
            theta_np = theta.detach().cpu().numpy().item()
            exp = circuit.run(theta_np)
            result = torch.tensor([exp], device=theta.device, dtype=theta.dtype)
            ctx.save_for_backward(theta, result)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            theta, _ = ctx.saved_tensors
            shift = ctx.shift
            theta_np = theta.detach().cpu().numpy().item()
            exp_plus = ctx.circuit.run(theta_np + shift)
            exp_minus = ctx.circuit.run(theta_np - shift)
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_tensor = torch.tensor([grad], device=theta.device, dtype=theta.dtype)
            return grad_tensor * grad_output, None, None

    def __init__(self, circuit: _TwoQubitVariationalCircuit, shift: float = np.pi/2):
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self._Func.apply(theta, self.circuit, self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """CNN backbone followed by a quantum variational head."""
    def __init__(self,
                 quantum_backend: str = 'aer',
                 quantum_shots: int = 100,
                 shift: float = np.pi/2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.circuit = _TwoQubitVariationalCircuit(shots=quantum_shots)
        self.head = _QuantumExpectationLayer(self.circuit, shift=shift)

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
        x = self.fc3(x)
        logits = x.squeeze(-1)
        probs = self.head(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
