"""Hybrid classical‑quantum convolutional network with 3‑qubit entangling circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile


class QuantumCircuit:
    """Parameterized 3‑qubit circuit with entanglement."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        for q in range(n_qubits):
            self._circuit.h(q)
        for q in range(n_qubits):
            self._circuit.ry(self.theta, q)
        self._circuit.cx(0, 1)
        self._circuit.cx(1, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        param_binds = [{self.theta: theta} for theta in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            probs = counts / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            n_qubits = self._circuit.num_qubits
            exp = 0.0
            for i in range(n_qubits):
                bits = ((states >> i) & 1).astype(int)
                exp += np.sum((1 - 2 * bits) * probs)
            return exp / n_qubits

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable quantum expectation via parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.squeeze().cpu().numpy()
        exp_vals = ctx.circuit.run(thetas)
        probs = (exp_vals + 1) / 2
        out = torch.tensor(probs, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        thetas = inputs.squeeze().cpu().numpy()
        shift = ctx.shift
        grads = []
        for theta in thetas:
            f_plus = ctx.circuit.run(np.array([theta + shift]))[0]
            f_minus = ctx.circuit.run(np.array([theta - shift]))[0]
            grads.append((f_plus - f_minus) / 4.0)
        grad_inputs = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype) * grad_output
        return grad_inputs, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards through a 3‑qubit circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)


class QCNet(nn.Module):
    """CNN followed by a 3‑qubit hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drop = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=3, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        res = x
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x += res
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.hybrid(x).squeeze()
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
