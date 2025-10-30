"""Quantum‑enhanced binary classifier that mirrors the classical surrogate.

This module implements the same `HybridBinaryClassifier` interface as the
classical version but replaces the filter and fully‑connected heads with
parameterised quantum circuits.  The design follows the same high‑level
architecture:

  * Convolutional backbone (identical to the classical version).
  * Quantum filter head that processes 2×2 image patches.
  * Quantum‑fully‑connected head that aggregates the filter outputs.
  * Differentiable hybrid sigmoid head implemented via a single‑qubit
    parameterised circuit.

Key components:
  * `SingleQubitCircuit` – a minimal circuit for computing a
    parameterised expectation value.
  * `PatchFilterCircuit` – a 4‑qubit circuit that encodes a 2×2 patch.
  * `HybridFunction` – differentiable interface between PyTorch and
    the quantum circuit using the parameter‑shift rule.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator


class SingleQubitCircuit:
    """Parameterised single‑qubit circuit for expectation evaluation."""

    def __init__(self, backend: AerSimulator, shots: int):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, theta: float) -> float:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta}],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        return np.sum(states * probs)


class PatchFilterCircuit:
    """4‑qubit circuit that processes a 2×2 image patch."""

    def __init__(self, backend: AerSimulator, shots: int, threshold: float):
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = 4
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(
            self.n_qubits, 2, skip_decompose=True
        )
        self.circuit.measure_all()

    def run(self, patch: np.ndarray) -> float:
        # patch shape (4,)
        param_binds = {}
        for i, val in enumerate(patch):
            param_binds[self.theta[i]] = np.pi if val > self.threshold else 0.0
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        return np.sum(states * probs)


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: SingleQubitCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # inputs: tensor of shape (batch,)
        expectations = []
        for val in inputs.detach().cpu().numpy():
            expectations.append(circuit.run(val))
        result = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            e_plus = ctx.circuit.run(val + shift)
            e_minus = ctx.circuit.run(val - shift)
            grad_inputs.append(e_plus - e_minus)
        grad_inputs = torch.tensor(grad_inputs, device=inputs.device, dtype=inputs.dtype)
        return grad_inputs * grad_output, None, None


class QuantumFilter(nn.Module):
    """Quantum filter head that processes 2×2 patches of the input image."""

    def __init__(self, threshold: float = 0.0, shots: int = 100):
        super().__init__()
        self.backend = AerSimulator()
        self.threshold = threshold
        self.shots = shots
        self.filter_circuit = PatchFilterCircuit(self.backend, shots, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        batch_size = x.shape[0]
        patches = []
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                patch = x[:, 0, i : i + 2, j : j + 2]  # (batch, 2, 2)
                patch_flat = patch.view(batch_size, -1).detach().cpu().numpy()
                expectations = []
                for p in patch_flat:
                    expectations.append(self.filter_circuit.run(p))
                patches.append(expectations)
        # patches list length 14*14, each element list of expectations per batch
        patch_tensor = torch.tensor(np.stack(patches, axis=1), device=x.device, dtype=x.dtype)
        # shape: (batch, 14*14)
        return patch_tensor.view(batch_size, 1, 14, 14)


class QuantumFullyConnected(nn.Module):
    """Quantum fully‑connected head that aggregates the filter outputs."""

    def __init__(self, shots: int = 100):
        super().__init__()
        self.backend = AerSimulator()
        self.shots = shots
        self.circuit = SingleQubitCircuit(self.backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_features)
        expectations = []
        for sample in x.detach().cpu().numpy():
            sample_expectations = []
            for theta in sample:
                sample_expectations.append(self.circuit.run(theta))
            expectations.append(np.mean(sample_expectations))
        return torch.tensor(expectations, device=x.device, dtype=x.dtype)


class HybridBinaryClassifier(nn.Module):
    """Quantum‑enhanced binary classifier that mirrors the classical surrogate."""

    def __init__(self, shift: float = np.pi / 2, threshold: float = 0.0, shots: int = 100) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Quantum filter head
        self.filter_head = QuantumFilter(threshold=threshold, shots=shots)

        # Quantum fully‑connected head
        self.fc_head = QuantumFullyConnected(shots=shots)

        # Hybrid sigmoid head
        self.shift = shift
        self.circuit = SingleQubitCircuit(AerSimulator(), shots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Quantum filter head
        x = self.filter_head(inputs)  # (batch, 1, 14, 14)

        # Convolutional backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Quantum fully‑connected head
        x = self.fc_head(x)  # (batch,)

        # Hybrid sigmoid head
        probs = HybridFunction.apply(x, self.circuit, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier", "HybridFunction"]
