"""Quantum‑augmented hybrid network combining a quantum quanvolution filter
and a differentiable quantum expectation head for binary classification.

The architecture mirrors the classical variant but replaces the quanvolution
filter with a circuit‑based kernel and the dense head with a quantum circuit
that outputs a probability via expectation value of Pauli‑Z.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
import torchquantum as tq


class QuantumCircuit:
    """Two‑qubit parameterised circuit executed on Aer simulator."""

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
    """Differentiable bridge between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit

        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Layer that feeds activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 image patches."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridQuanvolutionNet(nn.Module):
    """
    Quantum‑augmented hybrid network that mirrors the classical
    HybridQuanvolutionNet but replaces the quanvolution filter with
    a quantum kernel and the dense head with an expectation‑based quantum head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional branch (identical to the classical model)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Quantum quanvolution filter branch
        self.qfilter = QuantumQuanvolutionFilter()

        # Fully‑connected head
        self.fc1 = nn.Linear(1324, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(1, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional branch
        y = F.relu(self.conv1(x))
        y = self.pool(y)
        y = self.drop1(y)
        y = F.relu(self.conv2(y))
        y = self.pool(y)
        y = self.drop1(y)
        y = torch.flatten(y, 1)

        # Quantum quanvolution branch
        q = self.qfilter(x.mean(dim=1, keepdim=True))

        # Combined feature vector
        combined = torch.cat([y, q], dim=1)

        # Fully‑connected layers
        out = F.relu(self.fc1(combined))
        out = self.drop2(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # Quantum expectation head
        x = self.hybrid(out).T
        return torch.cat([x, 1 - x], dim=-1)


__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "QuantumQuanvolutionFilter",
    "HybridQuanvolutionNet",
]
