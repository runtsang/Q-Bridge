"""Hybrid quantum‑classical binary classifier.

Features a CNN feature extractor, a classical transformer encoder, and a
parameterised Qiskit circuit as the expectation head.  The quantum head
is differentiable via a custom autograd.Function that implements a
parameter‑shift derivative.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

# Quantum circuit used for the expectation head
class QuantumCircuit:
    """Two‑qubit parameterised circuit returning the <Z> expectation."""
    def __init__(self, n_qubits: int, backend, shots: int):
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
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards to a Qiskit circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        theta_vals = inputs.tolist()
        exp_vals = ctx.circuit.run(theta_vals)
        output = torch.tensor(exp_vals, device=inputs.device)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.tolist():
            exp_r = ctx.circuit.run([val + shift])[0]
            exp_l = ctx.circuit.run([val - shift])[0]
            grad_inputs.append(exp_r - exp_l)
        grad_inputs = torch.tensor(grad_inputs, device=grad_output.device)
        return grad_inputs * grad_output, None, None

class QuantumHybridHead(nn.Module):
    """Hybrid head that maps a scalar to a probability via a quantum circuit."""
    def __init__(self, n_qubits: int, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: [batch]
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# Classical transformer components reused from the ML module
class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridBinaryClassifier(nn.Module):
    """CNN + transformer encoder + quantum expectation head."""
    def __init__(self,
                 use_transformer: bool = True,
                 transformer_blocks: int = 1,
                 num_heads: int = 4,
                 ffn_dim: int = 128,
                 n_qubits: int = 2,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        # Feature extractor (same as classical)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(12 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Transformer
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(84, num_heads, ffn_dim) for _ in range(transformer_blocks)]
            )
        else:
            self.transformer = None

        # Quantum head
        self.quantum_head = QuantumHybridHead(n_qubits, shots, shift)

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
        x = self.fc3(x)

        if self.use_transformer:
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)

        probs = self.quantum_head(x.squeeze(-1))
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
