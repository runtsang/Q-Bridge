"""Quantum hybrid model based on Quantum‑NAT and binary classification.

The network uses the same convolutional backbone as the classical
variant but replaces the final linear projection with a parameterised
four‑qubit quantum circuit executed on Aer.  The expectation value of
Pauli‑Z on each qubit is returned as the class probabilities, and a
differentiable autograd wrapper (HybridFunction) allows end‑to‑end training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuitWrapper:
    """Parameterised four‑qubit circuit executed on the Aer simulator."""
    def __init__(self, n_qubits: int = 4, shots: int = 512):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple variational ansatz: H on all qubits, then Ry(theta) on each
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of parameter vectors.

        Parameters
        ----------
        params : np.ndarray of shape (batch, n_qubits)
            Each row contains the angles for the Ry gates.

        Returns
        -------
        expectations : np.ndarray of shape (batch, n_qubits)
            Expectation value <Z> for each qubit.
        """
        batch_expectations = []
        for angles in params:
            compiled = transpile(self.circuit, self.backend)
            job = self.backend.run(
                assemble(
                    compiled,
                    shots=self.shots,
                    parameter_binds=[{self.theta: a} for a in angles],
                )
            )
            result = job.result()
            counts = result.get_counts()
            expectations = []
            for qubit in range(self.n_qubits):
                # Build measurement string for qubit `qubit`
                exp = 0.0
                for state, count in counts.items():
                    z = 1 if state[-(qubit+1)] == "0" else -1
                    exp += z * count
                exp /= self.shots
                expectations.append(exp)
            batch_expectations.append(expectations)
        return np.array(batch_expectations)


class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the circuit
        input_np = inputs.detach().cpu().numpy()
        expectation = circuit.run(input_np)
        out = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for x in inputs.detach().cpu().numpy():
            # central difference for each qubit
            e_plus = ctx.circuit.run(np.array([x + shift]))
            e_minus = ctx.circuit.run(np.array([x - shift]))
            grads.append((e_plus - e_minus) / (2 * shift))
        grad_inputs = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grad_inputs * grad_output, None, None


class HybridLayer(nn.Module):
    """Quantum expectation head."""
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits=n_qubits, shots=512)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class QuantumNATHybrid(nn.Module):
    """Hybrid convolutional network with a quantum expectation head."""
    def __init__(self, in_channels: int = 1, n_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor mimicking the QML reference
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )
        self.norm = nn.BatchNorm1d(n_classes)
        self.hybrid = HybridLayer(n_qubits=n_classes, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.norm(x)
        # Quantum expectation head
        q = self.hybrid(x)
        # Return probability vector (softmax is applied externally if needed)
        return torch.cat([q, 1 - q], dim=-1) if q.ndim == 1 else q


__all__ = ["QuantumNATHybrid"]
