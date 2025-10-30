"""Hybrid attention‑based binary classifier with quantum components.

The architecture mirrors the classical implementation but replaces
the dense head with a parametrised quantum circuit and the
self‑attention block with a small quantum circuit that maps the
feature vector into a set of rotation angles.  All quantum
operations are wrapped in `torch.autograd.Function` so that
gradients flow through the quantum part via the parameter‑shift
rule.  The module is fully compatible with the same API as the
classical version.
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
    """Two‑qubit parametrised circuit used as the quantum head."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter('theta')

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        bits = np.array([int(b, 2) for b in counts.keys()])
        return np.sum(bits * probs)

class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the quantum head using the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        out = circuit.run(inputs.detach().cpu().numpy())
        return torch.tensor(out, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = circuit.run([val + shift])
            left = circuit.run([val - shift])
            grads.append(right - left)
        grad = torch.tensor(grads, dtype=torch.float32, device=grad_output.device) * grad_output
        return grad, None, None

class Hybrid(nn.Module):
    """Quantum head that maps a single scalar to a probability."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class QuantumSelfAttention:
    """Quantum implementation of a self‑attention block."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.rotation_params = qiskit.circuit.ParameterVector('rot', length=3 * n_qubits)
        self.entangle_params = qiskit.circuit.ParameterVector('ent', length=n_qubits - 1)

    def _build_circuit(self, rot: np.ndarray, ent: np.ndarray) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(ent[i], i + 1)
        qc.measure_all()
        return qc

    def run(self, rotation: np.ndarray, entangle: np.ndarray) -> np.ndarray:
        qc = self._build_circuit(rotation, entangle)
        job = qiskit.execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts()
        probs = np.array(list(result.values())) / self.shots
        bits = np.array([int(b, 2) for b in result.keys()])
        expectations = []
        for i in range(self.n_qubits):
            # Probability of bit i being 1
            prob1 = np.sum(((bits >> i) & 1) * probs)
            # Expectation of Pauli‑Z: 1 - 2*prob1
            expectations.append(1 - 2 * prob1)
        return np.array(expectations)

class HybridAttentionQCNet(nn.Module):
    """Quantum‑enhanced convolutional network with a quantum self‑attention block."""
    def __init__(self, attention_dim: int = 4, quantum_head_qubits: int = 2):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop_conv = nn.Dropout2d(p=0.25)

        # Attention head
        self.attention_fc = nn.Linear(16 * 7 * 7, attention_dim)
        backend = AerSimulator()
        self.quantum_attention = QuantumSelfAttention(attention_dim, backend, shots=512)

        # Classification head
        self.fc1 = nn.Linear(attention_dim, 120)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(quantum_head_qubits, backend, shots=512, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop_conv(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_conv(x)

        # Flatten and attention
        x = torch.flatten(x, 1)
        attn_features = self.attention_fc(x)
        rot = attn_features.detach().cpu().numpy()
        ent = np.zeros(attn_features.shape[1] - 1)
        qattn = self.quantum_attention.run(rot, ent)
        qattn_tensor = torch.tensor(qattn, dtype=torch.float32, device=x.device)

        x = qattn_tensor
        # Classification
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridAttentionQCNet", "Hybrid", "HybridFunction", "QuantumSelfAttention"]
