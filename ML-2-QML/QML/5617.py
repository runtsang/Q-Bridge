from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import qiskit
from qiskit import assemble, transpile

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with a transformer backbone."""
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention used in the classical backbone."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """Two‑layer feed‑forward block used inside each transformer."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block combining attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumCircuit:
    """Parameterized circuit used by the hybrid head.  The circuit
    accepts a single rotation angle per execution and returns the
    expectation of Pauli‑Z."""
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
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Custom autograd function that forwards to the quantum circuit
    and implements the parameter‑shift rule for gradients."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.detach().cpu().numpy().tolist())
        result = torch.tensor(expectation_z, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.numel()):
            val = inputs.flatten()[i].item()
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grad = right - left
            grad_inputs.view(-1)[i] = grad
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a scalar through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class EstimatorQNN(nn.Module):
    """
    Quantum‑enhanced estimator that mirrors the classical backbone
    but replaces the final head with a hybrid quantum expectation.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        dropout: float = 0.1,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.head = Hybrid(1, backend, shots=100, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, feature)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for block in self.transformer:
            x = block(x)
        x = self.norm(x.mean(dim=1))
        x = self.fc(x)
        return self.head(x)

__all__ = ['EstimatorQNN', 'Hybrid', 'HybridFunction', 'QuantumCircuit']
