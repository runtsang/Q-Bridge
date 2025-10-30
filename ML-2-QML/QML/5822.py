"""Quantum hybrid self‑attention binary classifier.

This module mirrors the classical architecture but replaces the
self‑attention block and the final head with quantum circuits.
The design demonstrates how a classical convolutional backbone
can be coupled to variational quantum circuits, enabling side‑by‑side
experiments on quantum‑classical scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile, assemble, Aer

class QuantumSelfAttention(nn.Module):
    """Variational self‑attention block implemented with a small
    multi‑qubit circuit.  Parameters are learned via gradient
    descent through the quantum expectation value.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq, embed_dim)
        batch, seq, embed = x.shape
        out = []
        for i in range(seq):
            # Encode the i‑th element as rotation angles
            angles = x[:, i, :].reshape(-1).numpy()
            rotation = np.zeros(3 * self.n_qubits)
            entangle = np.zeros(self.n_qubits - 1)
            circuit = self._build_circuit(rotation, entangle)
            job = execute(circuit, self.backend, shots=256)
            result = job.result().get_counts(circuit)
            probs = np.array(list(result.values()), dtype=float) / 256
            out.append(torch.tensor(probs, dtype=torch.float32))
        out = torch.stack(out, dim=1)  # (batch, seq)
        # Expand back to embed_dim dimension
        out = out.unsqueeze(-1).repeat(1, 1, embed)
        return out

class QuantumCircuitWrapper(nn.Module):
    """Simple two‑qubit variational circuit used as the final head."""
    def __init__(self, n_qubits: int, shots: int = 512):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        thetas_np = thetas.detach().cpu().numpy()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas_np])
        result = self.backend.run(qobj).result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        return torch.tensor([expectation(result)], dtype=torch.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit(inputs)
        out = expectation
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.circuit(torch.tensor([val + shift]))
            left = ctx.circuit(torch.tensor([val - shift]))
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum hybrid layer that maps a scalar to a probability."""
    def __init__(self, n_qubits: int, shots: int = 512, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class HybridSelfAttentionQCNet(nn.Module):
    """End‑to‑end hybrid network combining convolution, quantum self‑attention
    and a quantum head.  The architecture mirrors the classical version,
    enabling direct comparison of performance and scaling.
    """
    def __init__(self, embed_dim: int = 4, n_qubits_attn: int = 4,
                 n_qubits_head: int = 2, shots: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy = torch.zeros(1, 3, 32, 32)
        dummy = self._forward_features(dummy)
        feat_dim = dummy.shape[-1]

        self.self_attn = QuantumSelfAttention(n_qubits_attn)
        self.fc1 = nn.Linear(feat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits_head, shots)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        seq = x.shape[-1] // self.self_attn.n_qubits
        x = x.view(-1, seq, self.self_attn.n_qubits)
        x = self.self_attn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = self.hybrid(x)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["HybridSelfAttentionQCNet"]
