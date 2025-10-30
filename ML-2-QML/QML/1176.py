import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """
    Two‑qubit variational circuit with a trainable rotation angle.
    Returns the expectation value of Z on the first qubit.
    """
    def __init__(self, backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, theta_val: float) -> float:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta_val}]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        exp = 0.0
        for outcome, count in result.items():
            # outcome string is little‑endian; last bit is qubit 0
            z = 1 if outcome[-1] == "0" else -1
            exp += z * count
        return exp / self.shots

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards the classical output through
    the variational circuit and implements the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp_vals = [circuit.run(float(v.item())) for v in inputs]
        return torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for v in grad_output:
            exp_plus = circuit.run(float(v.item() + shift))
            exp_minus = circuit.run(float(v.item() - shift))
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards the classical activations through a
    variational quantum circuit.
    """
    def __init__(self, backend, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalQuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs)
        return HybridFunction.apply(squeezed, self.circuit, self.shift)

class MultiHeadAttentionEncoder(nn.Module):
    """
    Lightweight transformer encoder that aggregates spatial features.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

class HybridNet(nn.Module):
    """
    Convolutional network followed by a variational quantum head.
    Mirrors the classical HybridNet but replaces the linear head with
    a quantum circuit that learns a trainable rotation angle.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, shots: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        self.proj = nn.Linear(15 * 6 * 6, embed_dim)
        self.encoder = MultiHeadAttentionEncoder(embed_dim, num_heads)

        self.classifier = nn.Linear(embed_dim, 1)

        backend = AerSimulator()
        self.hybrid = Hybrid(backend, shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        x = x.unsqueeze(0)  # (1, batch, embed_dim)
        x = self.encoder(x)
        x = x.squeeze(0)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        probs = probs.unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridNet", "Hybrid", "HybridFunction", "VariationalQuantumCircuit"]
