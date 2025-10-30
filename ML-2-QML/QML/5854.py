"""Hybrid classical-quantum binary classifier.

This module defines QuantumHybridBinaryClassifier that uses a variational
quantum circuit as the prediction head. The backbone is identical to the
classical version. The quantum circuit is implemented with Qiskit Aer
simulator and is differentiable via a custom autograd function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile

class QuantumCircuit:
    """Parameterized two‑qubit variational circuit."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple entangling circuit
        for q in range(n_qubits):
            self.circuit.h(q)
        for q in range(n_qubits - 1):
            self.circuit.cx(q, q + 1)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameters."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        expectations = circuit.run(thetas)
        out = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = ctx.circuit.run([val + shift])[0]
            exp_minus = ctx.circuit.run([val - shift])[0]
            grad_inputs.append((exp_plus - exp_minus) / 2.0)
        grad_inputs = torch.tensor(grad_inputs, device=inputs.device, dtype=inputs.dtype)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum head that forwards activations through a variational circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x):
        return HybridFunction.apply(x, self.circuit, self.shift)

class QuantumHybridBinaryClassifier(nn.Module):
    """CNN backbone + quantum expectation head for binary classification."""
    def __init__(self,
                 in_channels: int = 3,
                 conv_channels: list | None = None,
                 hidden_dims: list[int] = [120, 84],
                 dropout_probs: tuple[float, float] = (0.2, 0.5),
                 backend=None,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        if conv_channels is None:
            conv_channels = [6, 15]
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=5,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3,
                               stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=dropout_probs[0])
        self.drop2 = nn.Dropout2d(p=dropout_probs[1])
        dummy_input = torch.zeros(1, in_channels, 32, 32)
        with torch.no_grad():
            x = self._forward_conv(dummy_input)
            flat_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        if backend is None:
            backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots, shift)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).squeeze()
        probs = self.hybrid(logits).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridBinaryClassifier"]
