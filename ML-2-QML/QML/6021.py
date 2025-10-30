"""
HybridClassifier – Quantum‑classical hybrid network.

This module implements a two‑qubit variational circuit with a
parameter‑shift gradient rule.  The classical CNN processes the
image, then the last fully‑connected output is fed into a quantum
circuit that returns an expectation value.  A small calibration
network maps this expectation to a probability.  A minimal test
harness demonstrates a forward pass.
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
    """
    A two‑qubit parameterised circuit executed on the Aer simulator.
    The circuit consists of an H on each qubit, a controlled‑X to
    entangle, and a single‑qubit Ry rotation whose angle is the
    classical feature.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        # Entangling block
        self._circuit.h(all_qubits)
        self._circuit.cx(0, 1)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the supplied angles and return the
        expectation value of the first qubit in the Z basis.
        """
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict: dict[str, int]) -> float:
            counts = np.array(list(count_dict.values()))
            probs = counts / self.shots
            # Convert bit strings to integers; the first qubit is the
            # most significant bit in the string.
            states = np.array([int(k, 2) for k in count_dict.keys()])
            # Z expectation of first qubit: (-1)^bit
            first_bits = (states >> (self._circuit.num_qubits - 1)) & 1
            z_vals = 1 - 2 * first_bits  # 0 -> +1, 1 -> -1
            return float(np.sum(z_vals * probs))

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that forwards the input through the quantum
    circuit and provides a parameter‑shift gradient in the backward
    pass.  The shift is a small constant (π/2) that guarantees
    exact gradients for a single‑parameter circuit.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the circuit for each input value
        thetas = inputs.detach().cpu().numpy()
        expectations = circuit.run(thetas)
        out = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        grad_in = torch.zeros_like(inputs)
        for i, theta in enumerate(thetas):
            # Parameter‑shift rule: (f(x+shift) - f(x-shift)) / (2*shift)
            f_plus = ctx.circuit.run(np.array([theta + shift]))[0]
            f_minus = ctx.circuit.run(np.array([theta - shift]))[0]
            grad = (f_plus - f_minus) / (2 * shift)
            grad_in[i] = grad
        return grad_in * grad_output, None, None


class QuantumHead(nn.Module):
    """
    Quantum head that maps a scalar feature to a probability.  The
    expectation value produced by the circuit is passed through a
    small linear calibration layer and a sigmoid.
    """
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift
        self.calibration = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be of shape (batch,)
        expectation = HybridFunction.apply(x, self.quantum_circuit, self.shift)
        logits = self.calibration(expectation.unsqueeze(-1)).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


class HybridClassifier(nn.Module):
    """
    Classical CNN followed by a quantum head.  The network mirrors the
    original QCNet architecture but replaces the classical head with
    a variational circuit that returns a calibrated probability.
    """
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
        self.quantum_head = QuantumHead(
            n_qubits=2,
            backend=backend,
            shots=200,
            shift=np.pi / 2,
        )

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
        x = self.fc3(x).squeeze(-1)
        probs = self.quantum_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


# Simple test harness
if __name__ == "__main__":
    model = HybridClassifier()
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)


__all__ = ["HybridClassifier", "QuantumHead", "HybridFunction", "QuantumCircuit"]
