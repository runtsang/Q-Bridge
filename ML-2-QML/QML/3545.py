"""Hybrid quantum‑classical binary classifier with quantum LSTM gates.

This module implements the same high‑level API as the classical
implementation but replaces the dense LSTM with a small quantum
circuit per gate.  The quantum head uses a single‑qubit expectation
value to produce the final class probability.  The code is fully
differentiable via a custom autograd function that approximates
gradients with finite differences.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble

# --------------------------------------------------------------------------- #
# 1. Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parametrised single‑qubit circuit that returns the expectation of Z."""
    def __init__(self, backend, shots: int = 200):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()
    def run(self, thetas: np.ndarray) -> np.ndarray:
        expectations = []
        for theta in thetas:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: theta}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            prob0 = counts.get("0", 0) / self.shots
            prob1 = counts.get("1", 0) / self.shots
            expectations.append(prob0 - prob1)
        return np.array(expectations)

# --------------------------------------------------------------------------- #
# 2. Differentiable interface
# --------------------------------------------------------------------------- #
class QuantumExpectationFunction(torch.autograd.Function):
    """Autograd wrapper that forwards the circuit expectation and
    estimates gradients with a central finite difference."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        theta = inputs.detach().cpu().numpy().flatten()
        expectations = circuit.run(theta).reshape(inputs.shape)
        out = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grad_inputs = []
        for val in inputs.detach().cpu().numpy().flatten():
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grad = (right - left) / (2 * shift)
            grad_inputs.append(grad)
        grad_inputs = np.array(grad_inputs).reshape(inputs.shape)
        grad_tensor = torch.tensor(grad_inputs, device=inputs.device, dtype=inputs.dtype)
        return grad_tensor * grad_output, None, None

# --------------------------------------------------------------------------- #
# 3. Gate module
# --------------------------------------------------------------------------- #
class QuantumGate(nn.Module):
    """Linear mapping followed by a quantum expectation."""
    def __init__(self, in_dim: int, hidden_dim: int, circuit: QuantumCircuit, shift: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.circuit = circuit
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.linear(x)
        return QuantumExpectationFunction.apply(angles, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
# 4. Quantum LSTM
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """LSTM with quantum gates implemented via 1‑qubit circuits."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 circuit: QuantumCircuit, shift: float = np.pi / 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        in_dim = input_dim + hidden_dim
        self.forget = QuantumGate(in_dim, hidden_dim, circuit, shift)
        self.input_gate = QuantumGate(in_dim, hidden_dim, circuit, shift)
        self.update = QuantumGate(in_dim, hidden_dim, circuit, shift)
        self.output_gate = QuantumGate(in_dim, hidden_dim, circuit, shift)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, input_dim)
        batch, seq_len, _ = inputs.shape
        hx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        for t in range(seq_len):
            x = inputs[:, t, :]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
        return hx  # final hidden state

# --------------------------------------------------------------------------- #
# 5. Quantum head (quantum expectation)
# --------------------------------------------------------------------------- #
class QuantumHead(nn.Module):
    """Linear layer followed by a quantum expectation head."""
    def __init__(self, in_features: int, circuit: QuantumCircuit, shift: float = np.pi / 2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.circuit = circuit
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.linear(x)
        probs = QuantumExpectationFunction.apply(angles, self.circuit, self.shift)
        return torch.cat([probs, 1 - probs], dim=1)

# --------------------------------------------------------------------------- #
# 6. CNN Backbone (identical to classical version)
# --------------------------------------------------------------------------- #
class CNNBackbone(nn.Module):
    """Convolutional feature extractor that outputs a sequence of scalars."""
    def __init__(self, seq_len: int = 50) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, seq_len)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        return x

# --------------------------------------------------------------------------- #
# 7. Full hybrid classifier (quantum)
# --------------------------------------------------------------------------- #
class HybridQuantumBinaryClassifier(nn.Module):
    """Quantum‑enhanced binary classifier mirroring the classical API."""
    def __init__(self, seq_len: int = 50, lstm_hidden_dim: int = 32,
                 backend=None, shots: int = 200, shift: float = np.pi / 2):
        super().__init__()
        self.backbone = CNNBackbone(seq_len=seq_len)
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        circuit = QuantumCircuit(backend, shots=shots)
        self.lstm = QuantumQLSTM(input_dim=1, hidden_dim=lstm_hidden_dim,
                                 circuit=circuit, shift=shift)
        self.head = QuantumHead(in_features=lstm_hidden_dim, circuit=circuit, shift=shift)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (batch, seq_len)
        seq = features.unsqueeze(-1)        # (batch, seq_len, 1)
        hidden = self.lstm(seq)             # (batch, hidden_dim)
        probs = self.head(hidden)           # (batch, 2)
        return probs

__all__ = ["HybridQuantumBinaryClassifier"]
