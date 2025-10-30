"""Quantum hybrid network for binary classification.

The module exposes:
* QuantumCircuit – parametrised 2‑qubit circuit.
* HybridFunction – differentiable bridge between PyTorch and the quantum backend.
* Hybrid – quantum expectation head.
* SamplerHybrid – sampler head using Qiskit Machine Learning's SamplerQNN.
* QCNet – CNN feature extractor followed by one of the quantum heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


class QuantumCircuit:
    """Two‑qubit circuit with a single parameter theta applied to all qubits."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for value in inputs.tolist():
            right = ctx.quantum_circuit.run([value + shift])
            left = ctx.quantum_circuit.run([value - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Quantum expectation head."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        exp = HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)
        exp = exp.unsqueeze(-1)
        return torch.cat([exp, 1 - exp], dim=-1)


class SamplerHybrid(nn.Module):
    """Sampler‑based head that uses Qiskit’s SamplerQNN to generate a 2‑class probability vector."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        super().__init__()
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)
        qc = qiskit.QuantumCircuit(n_qubits)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        qc.measure_all()

        sampler = Sampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )
        self.backend = backend
        self.shots = shots

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = []
        for sample in inputs:
            param_binds = {'x[0]': float(sample[0].item()), 'x[1]': float(sample[1].item())}
            result = self.sampler_qnn.run(param_binds, shots=self.shots)
            counts = result.get_counts()
            total = sum(counts.values())
            probs.append([counts.get('0', 0) / total, counts.get('1', 0) / total])
        return torch.tensor(probs, dtype=torch.float32)


class QCNet(nn.Module):
    """CNN feature extractor followed by a quantum head (expectation or sampler)."""
    def __init__(self, head: str = "expectation") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        backend = qiskit.Aer.get_backend("aer_simulator")

        if head == "expectation":
            self.head = Hybrid(1, backend, shots=100, shift=np.pi / 2)
        elif head == "sampler":
            self.head = SamplerHybrid(2, backend, shots=100)
        else:
            raise ValueError(f"Unsupported head type {head!r}")

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
        logits = self.fc3(x)
        probs = self.head(logits)
        return probs


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "SamplerHybrid", "QCNet"]
