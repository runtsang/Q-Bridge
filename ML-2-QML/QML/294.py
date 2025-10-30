"""Quantum‑enhanced classifier with a 4‑qubit variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class ParametricQuantumCircuit:
    """Four‑qubit variational circuit with entanglement layers."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.ParameterVector("theta", n_qubits)
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        # Rotation layers
        for q in range(n_qubits):
            self._circuit.ry(self.theta[q], q)

        # Entanglement
        for q in range(n_qubits - 1):
            self._circuit.cx(q, q + 1)
        self._circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        """Execute the circuit and return the mean Z‑expectation of the first qubit."""
        bind_dict = {self.theta[i]: float(params[i]) for i in range(self.n_qubits)}
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind_dict])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for state, count in counts.items():
            z = 1 if state[0] == "0" else -1
            exp += z * count
        return exp / self.shots


class HybridFunction(torch.autograd.Function):
    """Differentiable interface using the parameter‑shift rule."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: ParametricQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp_z = ctx.circuit.run(inputs.detach().cpu().numpy())
        return torch.tensor([exp_z], dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors if hasattr(ctx, "saved_tensors") else (None, None)
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.cpu().numpy():
            grads.append(circuit.run([val + shift]) - circuit.run([val - shift]))
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer forwarding activations through the variational circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = ParametricQuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The quantum circuit expects a 1‑D vector of qubit parameters.
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


class QCNet(nn.Module):
    """CNN followed by a 4‑qubit variational quantum head for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.res1 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=False)  # placeholder for residual logic
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Flattened feature size for 32×32 RGB inputs
        self.fc1 = nn.Linear(735, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits=4, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        # Residual connection (identity) for simplicity
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Pass the scalar through the quantum hybrid layer
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["ParametricQuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
