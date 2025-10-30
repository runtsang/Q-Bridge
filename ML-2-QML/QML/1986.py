import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum utilities – based on TorchQuantum
# --------------------------------------------------------------------------- #
class QuantumHead(tq.QuantumModule):
    """Per‑head quantum circuit producing d‑dimensional output."""
    def __init__(self, d_k: int, depth: int = 1):
        super().__init__()
        self.n_wires = d_k
        self.depth = depth
        # Encode classical data into rotation angles
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
        )
        # Parameterized gates per depth
        self.params = nn.ModuleList(
            [nn.Sequential(*[tq.RX(has_params=True, trainable=True) for _ in range(d_k)]) for _ in range(depth)]
        )
        # Entangling layer
        self.entangle = nn.ModuleList(
            [nn.Sequential(*[tqf.cnot(q_device=None, wires=[i, (i+1) % d_k]) for i in range(d_k)]) for _ in range(depth)
        ])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for layer in self.params:
            layer(q_device)
        for layer in self.entangle:
            layer(q_device, q_device)
        return self.measure(q_device)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 quantum_depth: int = 1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.head_circuits = nn.ModuleList([
            QuantumHead(self.d_k, depth=quantum_depth) for _ in range(num_heads)
        ])
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply per‑head quantum circuits
        for i in range(self.num_heads):
            q[:, i] = self.head_circuits[i](q[:, i], self.q_device)
            k[:, i] = self.head_circuits[i](k[:, i], self.q_device)
            v[:, i] = self.head_circuits[i](v[:, i], self.q_device)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class FeedForwardQuantum(tq.QuantumModule):
    """Quantum feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, depth: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList(
            [nn.Sequential(*[tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]) for _ in range(depth)]
        )
        self.entangle = nn.ModuleList(
            [nn.Sequential(*[tqf.cnot(q_device=None, wires=[i, (i+1) % n_qubits]) for i in range(n_qubits)]) for _ in range(depth)
        ])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :].view(batch, -1)
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=token.device)
            self.encoder(qdev, token)
            for layer in self.params:
                layer(qdev)
            for layer in self.entangle:
                layer(qdev, qdev)
            outputs.append(self.measure(qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(F.relu(out)))
        return self.linear2(out)


class TransformerBlockQuantum(nn.Module):
    """Transformer block with quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 quantum_depth: int = 1,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              quantum_depth=quantum_depth,
                                              q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, depth=n_qlayers)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encodings – unchanged."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class QuantumHybridTransformer(nn.Module):
    """Transformer‑based text classifier with optional quantum blocks."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 quantum_depth: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        q_device = q_device or tq.QuantumDevice(n_wires=max(n_qubits_transformer, n_qubits_ffn))
        for _ in range(num_blocks):
            block = TransformerBlockQuantum(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                quantum_depth=quantum_depth,
                n_qubits_ffn=n_qubits_ffn,
                n_qlayers=n_qlayers,
                q_device=q_device,
                dropout=dropout
            )
            self.blocks.append(block)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumHybridTransformer",
]
