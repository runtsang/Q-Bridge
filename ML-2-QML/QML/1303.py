"""Quantum‑enhanced transformer implemented with PennyLane.

The class `QuantumTransformer` mirrors the public API of the
original `TextClassifier` but replaces the transformer blocks with
quantum‑aware variants.  Each block contains a variational quantum
attention module and a variational quantum feed‑forward module.
The implementation runs on the local `default.qubit` simulator and
does not require external hardware.

The quantum circuits are intentionally small – a single‑qubit or
few‑qubit circuit per head – so that the model can be trained on a
CPU.  The circuits are parameterised and differentiable, allowing
end‑to‑end training with PyTorch optimisers.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumAttentionLayer(nn.Module):
    """Self‑attention with a variational quantum circuit per head."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qubits_per_head: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Classical projection layers
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum circuit parameters
        self.n_qubits = n_qubits_per_head or self.d_k
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        self.param = nn.Parameter(torch.randn(self.n_qubits))

        # Trainable weight that controls the blend of classical and quantum
        self.quantum_weight = nn.Parameter(torch.tensor(0.5))

        # Linear mapping if qubit count differs from d_k
        self.proj = (
            nn.Linear(self.n_qubits, self.d_k) if self.n_qubits!= self.d_k else None
        )

        # Build the qnode
        def circuit(x):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits):
                qml.RZ(self.param[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = qml.qnode(circuit, device=self.q_device, interface="torch")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()

        # Classical projections
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        classical_out = torch.matmul(attn_weights, v)  # (batch, heads, seq, d_k)

        # Quantum transformation of the query vectors
        q_flat = q.permute(0, 2, 1, 3).reshape(-1, self.d_k)  # (batch*heads*seq, d_k)
        quantum_out = []
        for vec in q_flat:
            quantum_out.append(self.qnode(vec))
        quantum_out = torch.stack(quantum_out, dim=0)  # (batch*heads*seq, n_qubits)
        if self.proj is not None:
            quantum_out = self.proj(quantum_out)
        quantum_out = quantum_out.reshape(batch, self.num_heads, seq, self.d_k).permute(
            0, 2, 1, 3
        )

        # Blend classical and quantum outputs
        weight = torch.sigmoid(self.quantum_weight)
        out = weight * quantum_out + (1 - weight) * classical_out

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class QuantumFeedForwardLayer(nn.Module):
    """Feed‑forward network realised by a small variational quantum circuit."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        n_qubits: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

        # Classical linear layers
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

        # Quantum circuit parameters
        self.n_qubits = n_qubits or embed_dim
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        self.param = nn.Parameter(torch.randn(self.n_qubits))

        # Trainable weight for blending
        self.quantum_weight = nn.Parameter(torch.tensor(0.5))

        # Linear mapping if qubit count differs from embed_dim
        self.proj = (
            nn.Linear(self.n_qubits, self.embed_dim) if self.n_qubits!= self.embed_dim else None
        )

        # Build qnode
        def circuit(x):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits):
                qml.RZ(self.param[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = qml.qnode(circuit, device=self.q_device, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical path
        out = self.linear1(x)
        classical = F.relu(out)

        # Quantum path
        batch, seq, _ = x.size()
        x_flat = x.view(-1, self.embed_dim)  # (batch*seq, embed_dim)
        quantum_out = []
        for vec in x_flat:
            quantum_out.append(self.qnode(vec))
        quantum_out = torch.stack(quantum_out, dim=0)  # (batch*seq, n_qubits)
        if self.proj is not None:
            quantum_out = self.proj(quantum_out)
        quantum_out = quantum_out.view(batch, seq, self.embed_dim)

        weight = torch.sigmoid(self.quantum_weight)
        out = weight * quantum_out + (1 - weight) * classical
        out = self.linear2(self.dropout(out))
        return out


class QuantumTransformerBlock(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        n_qubits_per_head: Optional[int] = None,
        n_qubits_ffn: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttentionLayer(
            embed_dim,
            num_heads,
            dropout,
            n_qubits_per_head=n_qubits_per_head,
        )
        self.ffn = QuantumFeedForwardLayer(
            embed_dim,
            ffn_dim,
            dropout,
            n_qubits=n_qubits_ffn,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class QuantumPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with an optional quantum transform."""

    def __init__(self, embed_dim: int, max_len: int = 5000, n_qubits: Optional[int] = None):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Quantum transform parameters
        self.n_qubits = n_qubits or embed_dim
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        self.param = nn.Parameter(torch.randn(self.n_qubits))
        self.quantum_weight = nn.Parameter(torch.tensor(0.5))
        self.proj = (
            nn.Linear(self.n_qubits, embed_dim) if self.n_qubits!= embed_dim else None
        )

        # Build qnode
        def circuit(x):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits):
                qml.RZ(self.param[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = qml.qnode(circuit, device=self.q_device, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.pe[:, : x.size(1)]
        batch, seq, _ = x.size()
        x_flat = out.view(-1, out.size(-1))
        quantum_out = []
        for vec in x_flat:
            quantum_out.append(self.qnode(vec))
        quantum_out = torch.stack(quantum_out, dim=0)
        if self.proj is not None:
            quantum_out = self.proj(quantum_out)
        quantum_out = quantum_out.view(batch, seq, out.size(-1))
        weight = torch.sigmoid(self.quantum_weight)
        return weight * quantum_out + (1 - weight) * out


class QuantumTransformer(nn.Module):
    """Quantum‑enhanced transformer text classifier.

    The class mirrors the original `TextClassifier` API but replaces
    the transformer blocks with quantum‑aware variants.  All quantum
    operations are implemented with PennyLane and run on the local
    `default.qubit` simulator, so the model can be trained on a CPU.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_per_head: Optional[int] = None,
        n_qubits_ffn: Optional[int] = None,
        n_qubits_pos: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = QuantumPositionalEncoder(
            embed_dim, n_qubits=n_qubits_pos
        )
        self.blocks = nn.ModuleList(
            [
                QuantumTransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    n_qubits_per_head=n_qubits_per_head,
                    n_qubits_ffn=n_qubits_ffn,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        mask = None  # no masking in this simple example
        for block in self.blocks:
            x = block(x, mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumAttentionLayer",
    "QuantumFeedForwardLayer",
    "QuantumTransformerBlock",
    "QuantumPositionalEncoder",
    "QuantumTransformer",
]
