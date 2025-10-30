"""HybridQuantumBinaryClassifier: Hybrid PyTorch model with a variational quantum head.

This module demonstrates a lightweight hybrid model that uses a two‑qubit
variational circuit to compute the final binary decision.  The circuit is
executed on Qiskit Aer simulator and gradients are propagated through a
parameter‑shift rule implemented in a custom torch.autograd.Function.
"""

import torch
import torch.nn as nn
import numpy as np
import qiskit

from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter
from qiskit.result import Result

class VariationalQuantumCircuit:
    """Two‑qubit variational circuit with a single parameter per input."""

    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        # Base circuit with a parameter
        self.theta = Parameter("θ")
        self.base_circuit = qiskit.QuantumCircuit(n_qubits)
        self.base_circuit.h(range(n_qubits))
        # Apply Ry(θ) on each qubit
        self.base_circuit.ry(self.theta, range(n_qubits))
        # Measure all qubits
        self.base_circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Run the circuit for a batch of θ values and return expectation of Z⊗Z."""
        results = []
        for theta in thetas:
            circ = self.base_circuit.copy()
            circ = transpile(circ, self.backend)
            circ = circ.bind_parameters({self.theta: theta})
            qobj = assemble(circ, shots=self.shots)
            job = self.backend.run(qobj)
            result: Result = job.result()
            counts = result.get_counts()
            # Compute expectation value of Z⊗Z
            exp = 0.0
            total = sum(counts.values())
            for bitstring, count in counts.items():
                # Map '0' -> +1, '1' -> -1
                z_vals = [1 if b == "0" else -1 for b in bitstring]
                z_prod = np.prod(z_vals)
                exp += z_prod * count / total
            results.append(exp)
        return np.array(results)

class QuantumExpectation(torch.autograd.Function):
    """Autograd wrapper that evaluates the variational circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(thetas)
        result = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        grad_inputs = []
        for theta in thetas:
            exp_plus = ctx.circuit.run(np.array([theta + shift]))[0]
            exp_minus = ctx.circuit.run(np.array([theta - shift]))[0]
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

class HybridLayer(nn.Module):
    """Layer that forwards a scalar through the variational circuit."""

    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalQuantumCircuit(n_qubits=n_qubits, backend=backend, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs shape (batch,)
        return QuantumExpectation.apply(inputs.squeeze(), self.circuit, self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """Hybrid CNN + variational quantum head for binary classification."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Feature extractor (same as classical backbone)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(15, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
        )
        # Hybrid quantum layer
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = HybridLayer(n_qubits=2, backend=backend, shots=1024, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)  # shape (batch,1)
        x = x.squeeze(-1)       # shape (batch,)
        x = self.hybrid(x)      # shape (batch,)
        x = x.unsqueeze(-1)     # shape (batch,1)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
