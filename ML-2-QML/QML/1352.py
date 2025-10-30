"""HybridQuantumConvNet – quantum‑classical hybrid with multi‑qubit variational circuit and attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuit:
    """Parameterized 2‑qubit variational circuit."""
    def __init__(self, backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        # Parameters
        self.theta0 = qiskit.circuit.Parameter("theta0")
        self.theta1 = qiskit.circuit.Parameter("theta1")
        # Ansatz
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta0, 0)
        self.circuit.ry(self.theta1, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for each row in params (shape: batch, 2)."""
        compiled = transpile(self.circuit, self.backend)
        batch = params.shape[0]
        qobjs = []
        for i in range(batch):
            bind = {self.theta0: params[i, 0], self.theta1: params[i, 1]}
            qobj = assemble(compiled, parameter_binds=[bind], shots=self.shots)
            qobjs.append(qobj)
        results = [self.backend.run(qobj).result() for qobj in qobjs]
        expectations = []
        for result in results:
            counts = result.get_counts()
            expectations.append(self._expectation_zz(counts))
        return np.array(expectations)

    @staticmethod
    def _expectation_zz(counts: dict) -> float:
        """Expectation value of Z⊗Z."""
        probs = {}
        for state, cnt in counts.items():
            probs[state] = cnt
        total = sum(probs.values())
        exp = 0.0
        for state, cnt in probs.items():
            prob = cnt / total
            # Z⊗Z eigenvalue: +1 if bits equal, -1 otherwise
            if state[0] == state[1]:
                exp += prob
            else:
                exp -= prob
        return exp


class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        batch = inputs.shape[0]
        grads = []
        for i in range(batch):
            param = inputs[i].detach().cpu().numpy()
            grad = []
            for d in range(2):
                shift_vec = np.zeros(2)
                shift_vec[d] = shift
                exp_plus = ctx.circuit.run((param + shift_vec).reshape(1, -1))[0]
                exp_minus = ctx.circuit.run((param - shift_vec).reshape(1, -1))[0]
                grad.append((exp_plus - exp_minus) / 2.0)
            grads.append(grad)
        grads = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        return grads * grad_output.squeeze(1).unsqueeze(1), None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the quantum circuit."""
    def __init__(self, backend, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class AttentionBlock(nn.Module):
    """Self‑attention over flattened spatial features."""
    def __init__(self, in_channels: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_channels, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = h * w
        x = x.view(b, c, seq).transpose(1, 2)  # (batch, seq, c)
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out + x
        attn_out = self.norm(attn_out)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return attn_out


class QCNet(nn.Module):
    """Convolutional network followed by a variational quantum head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.attn = AttentionBlock(15, heads=1)
        self.fc1 = nn.Linear(15 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        backend = AerSimulator()
        self.hybrid = Hybrid(backend, shots=1024, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.attn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        probs = torch.sigmoid(probs)
        return probs


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "AttentionBlock", "QCNet"]
