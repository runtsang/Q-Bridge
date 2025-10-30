"""Quantum‑classical hybrid binary classifier with batched, adaptive shot evaluation.

This module extends the original hybrid network by:
1. Allowing the circuit to accept a batch of rotation angles.
2. Supporting a variable number of shots per batch, which can be tuned
   during training for a trade‑off between accuracy and runtime.
3. Exposing a ``get_expectation`` API that returns a NumPy array of
   expectation values for a batch of parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuit:
    """Vectorised two‑qubit circuit with optional adaptive shot counts."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def get_expectation(self, thetas: np.ndarray, shots: int | None = None) -> np.ndarray:
        """Run the circuit for a batch of angles and return expectation values."""
        if shots is None:
            shots = self.shots

        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: theta} for theta in thetas]
        qobj = assemble(compiled, shots=shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        results = job.result().get_counts()

        def expectation(count_dict: dict[str, int]) -> float:
            counts = np.array(list(count_dict.values()))
            probs = counts / shots
            # Interpret measured bitstrings as integers (binary to decimal)
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.sum(states * probs)

        if isinstance(results, list):
            return np.array([expectation(r) for r in results])
        return np.array([expectation(results)])

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Convenience wrapper that uses the default shot count."""
        return self.get_expectation(thetas, shots=self.shots)


class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit using the parameter shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to a numpy array for the circuit
        angles = inputs.detach().cpu().numpy()
        exp_vals = circuit.get_expectation(angles, shots=circuit.shots)
        return torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit

        # Parameter‑shift rule: (f(x+shift)-f(x-shift))/(2*shift)
        angles = inputs.detach().cpu().numpy()
        exp_plus = circuit.get_expectation(angles + shift, shots=circuit.shots)
        exp_minus = circuit.get_expectation(angles - shift, shots=circuit.shots)
        grad = (exp_plus - exp_minus) / (2 * shift)

        return grad * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int = 1024,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure the input is a 1‑D tensor
        flat = torch.squeeze(inputs)
        return HybridFunction.apply(flat, self.circuit, self.shift)


class QCNet(nn.Module):
    """Convolutional network followed by a quantum‑expectation head."""
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
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=512,
                             shift=np.pi / 2)

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

        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
