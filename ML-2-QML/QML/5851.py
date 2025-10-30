"""Quantum hybrid binary classifier.

The quantum head is a two‑qubit variational circuit that encodes the
activations of the classical backbone as rotation angles.  The
expectation value of Pauli‑Z on the first qubit is passed through a
sigmoid to obtain a probability.  Back‑propagation is enabled via a
custom autograd function that implements the parameter‑shift rule.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Parameterized two‑qubit circuit executed on AerSimulator."""

    def __init__(self,
                 backend: qiskit.providers.Backend,
                 shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = Parameter("θ")

        # Universal gate set
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the supplied rotation angles."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        return self._expectation(result)

    def _expectation(self, counts: dict) -> np.ndarray:
        """Return the expectation value of Pauli‑Z on qubit 0."""
        probs = {k: v / self.shots for k, v in counts.items()}
        exp = 0.0
        for state, p in probs.items():
            z = 1 if state[0] == "0" else -1
            exp += z * p
        return np.array([exp])

class QuantumHybridFunction(torch.autograd.Function):
    """Autograd wrapper for the quantum circuit with parameter‑shift."""

    @staticmethod
    def forward(ctx,
                inputs: torch.Tensor,
                circuit: QuantumCircuitWrapper,
                shift: float = np.pi / 2) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.cpu().detach().numpy()
        exp = circuit.run(angles)
        out = torch.tensor(exp, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for val in inputs.cpu().detach().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grad.append((right - left) / 2.0)
        grad = torch.tensor(grad, dtype=torch.float32, device=grad_output.device)
        return grad * grad_output, None, None

class HybridBinaryClassifier(nn.Module):
    """Convolutional backbone + quantum expectation head."""

    def __init__(self,
                 in_channels: int = 3,
                 num_qubits: int = 2,
                 shots: int = 1024) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 1),
        )
        self.backend = AerSimulator()
        self.quantum_circuit = QuantumCircuitWrapper(self.backend, shots=shots)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        # Forward through quantum head
        probs = QuantumHybridFunction.apply(x.squeeze(), self.quantum_circuit, self.shift)
        probs = torch.sigmoid(probs)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier", "QuantumCircuitWrapper", "QuantumHybridFunction"]
