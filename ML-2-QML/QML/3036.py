"""Quantum hybrid binary classifier with a Qiskit expectation head.

The architecture mirrors the classical version but replaces the final
linear + sigmoid head with a parameter‑shifted quantum circuit
inspired by the Quantum‑NAT encoder.  The circuit is executed on
Aer's statevector simulator and returns the expectation of Pauli‑Z
on the first qubit.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Feature‑map circuit with a random layer."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]

        # Feature encoding: Ry rotations per qubit
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)

        # Random layer of 50 CX gates
        self.circuit.h(range(n_qubits))
        for _ in range(50):
            a, b = np.random.choice(n_qubits, 2, replace=False)
            self.circuit.cx(a, b)

        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter vectors."""
        expectations = []
        for theta in thetas:
            compiled = transpile(self.circuit, self.backend)
            job = self.backend.run(
                assemble(
                    compiled,
                    shots=self.shots,
                    parameter_binds=[{p: t} for p, t in zip(self.params, theta)],
                )
            )
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                bit = int(state[-1])  # first qubit is last bit
                exp += ((-1) ** bit) * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grads.append((right - left) / 2)
        grad_inputs = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grad_inputs * grad_output, None, None

class HybridQuantumLayer(nn.Module):
    """Hybrid layer that forwards activations through a Qiskit circuit."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        backend = AerSimulator()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """Quantum hybrid binary classifier mirroring the classical architecture."""
    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        # Convolutional feature extractor identical to the classical version
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 4),
            nn.BatchNorm1d(4),
        )
        self.quantum_head = HybridQuantumLayer(n_qubits=4, shots=1024, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        probs = self.quantum_head(x).squeeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier", "HybridFunction", "HybridQuantumLayer", "QuantumCircuitWrapper"]
