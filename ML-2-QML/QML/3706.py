"""AdvancedEstimatorQNN – quantum‑enhanced regression model.

The quantum part is implemented with Qiskit.  The model keeps the same
classical transformer encoder (adapted from the ML reference) and adds
a variational circuit that maps the first input feature to a
Pauli‑Z expectation value.  The quantum output is concatenated with
the classical latent vector and fed to a shared linear head.  The
module is fully PyTorch‑compatible and can be dropped into existing
pipelines.

Author: gpt-oss-20b
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum imports
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

# Classical transformer components (adapted from the ML reference)
class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = _FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class _PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AdvancedEstimatorQNN(nn.Module):
    """Quantum‑enhanced regression model that combines a transformer encoder
    with a variational circuit.  The model accepts a 2‑D input tensor and
    returns a single‑dimensional prediction."""
    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 8,
        num_heads: int = 2,
        ffn_dim: int = 4,
        dropout: float = 0.1,
        n_qubits: int = 1,
    ) -> None:
        super().__init__()
        # Classical part
        self.input_linear = nn.Linear(1, embed_dim)
        self.pos_encoder = _PositionalEncoder(embed_dim)
        self.transformer = _TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        # Quantum part
        # Define a single‑qubit variational circuit
        self.input_param = Parameter("theta1")
        self.weight_param = Parameter("theta2")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.input_param, 0)
        qc.rx(self.weight_param, 0)
        observable = SparsePauliOp.from_list([("Z", 1)])
        # Estimator
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

        # Final head that consumes both classical and quantum features
        self.head = nn.Linear(embed_dim + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extractor
        x_cl = x.unsqueeze(-1)  # (batch, 2, 1)
        x_cl = self.input_linear(x_cl)  # (batch, 2, embed_dim)
        x_cl = self.pos_encoder(x_cl)
        x_cl = self.transformer(x_cl)
        x_cl = x_cl.mean(dim=1)  # (batch, embed_dim)

        # Quantum feature extractor
        batch_size = x.shape[0]
        quantum_vals = []
        for i in range(batch_size):
            params = {self.input_param: float(x[i, 0]),
                      self.weight_param: float(0.0)}  # weight set to zero by default
            q_out = self.qnn.forward(params)
            quantum_vals.append(float(q_out["output"][0]))
        q_vec = torch.tensor(quantum_vals, dtype=x.dtype, device=x.device).unsqueeze(1)

        # Combine classical and quantum features
        combined = torch.cat([x_cl, q_vec], dim=1)  # (batch, embed_dim+1)
        return self.head(self.dropout(combined))


__all__ = ["AdvancedEstimatorQNN"]
