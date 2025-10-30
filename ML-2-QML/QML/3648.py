"""Hybrid attention network with a quantum self‑attention block.

The network is identical to the classical version but replaces the
classical attention and sigmoid head with a quantum circuit that
executes a self‑attention style sub‑circuit and measures the
expectation value of the first qubit.  The result is passed through a
sigmoid to obtain a probability distribution.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend: AerSimulator, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024) -> float:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result().get_counts(circuit)
        # Compute expectation of Pauli‑Z on the first qubit
        exp_z = 0.0
        for bits, count in result.items():
            bit = int(bits[-1])  # last bit corresponds to qubit 0
            prob = count / shots
            exp_z += ((-1) ** bit) * prob
        return exp_z

class HybridFunction(torch.autograd.Function):
    """Differentiable quantum head that returns the expectation of a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumSelfAttention, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Prepare parameters
        theta = inputs.squeeze().item()
        rotation_params = np.full((circuit.n_qubits * 3,), theta)
        entangle_params = np.full((circuit.n_qubits - 1,), theta)
        exp_z = ctx.circuit.run(Aer.get_backend("aer_simulator"),
                                rotation_params, entangle_params)
        result = torch.tensor([exp_z], dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        gradients = []
        val_np = inputs.detach().cpu().numpy()
        for val in val_np:
            right = ctx.circuit.run(Aer.get_backend("aer_simulator"),
                                    np.full((ctx.circuit.n_qubits * 3,), val + shift),
                                    np.full((ctx.circuit.n_qubits - 1,), val + shift))
            left = ctx.circuit.run(Aer.get_backend("aer_simulator"),
                                   np.full((ctx.circuit.n_qubits * 3,), val - shift),
                                   np.full((ctx.circuit.n_qubits - 1,), val - shift))
            gradients.append(right - left)
        gradients = torch.tensor([gradients], dtype=grad_output.dtype, device=grad_output.device)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum hybrid head that computes the expectation of a self‑attention circuit."""
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumSelfAttention(n_qubits)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridAttentionNet(nn.Module):
    """CNN → linear embedding → quantum self‑attention → fully‑connected → quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Linear embedding to 4‑dimensional space
        self.linear_embed = nn.Linear(55815, 4)

        # Quantum self‑attention block
        self.attention = QuantumSelfAttention(n_qubits=4)

        # Fully‑connected layers
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = Hybrid(n_qubits=4, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = self.linear_embed(x)          # (batch, 4)

        # Prepare parameters for the quantum self‑attention circuit
        exp_z_list = []
        for sample in x:
            sample_np = sample.detach().cpu().numpy()
            rotation_params = np.tile(sample_np, 3)
            entangle_params = np.tile(sample_np[:3], 1)
            exp_z = self.attention.run(Aer.get_backend("aer_simulator"),
                                       rotation_params, entangle_params)
            exp_z_list.append(exp_z)
        x = torch.tensor(exp_z_list, dtype=x.dtype, device=x.device).unsqueeze(1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

__all__ = ["QuantumSelfAttention", "HybridFunction", "Hybrid", "HybridAttentionNet"]
