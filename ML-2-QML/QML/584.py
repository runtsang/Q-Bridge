import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class MultiHeadAttentionQuantum(nn.Module):
    """Quantum multi‑head attention where each head is a small quantum circuit."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qubits_per_head: int = 2,
        q_device: Optional[tq.QuantumDevice] = None,
    ):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits_per_head, device='cpu')
        self.q_layer = self._build_q_layer(n_qubits_per_head)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _build_q_layer(self, n_wires: int):
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, dev: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(dev, x)
                for w, gate in enumerate(self.params):
                    gate(dev, wires=w)
                return self.measure(dev)
        return QLayer()

    def _quantum_head(self, tensor: torch.Tensor) -> torch.Tensor:
        B, T, H = tensor.shape
        dev = self.q_device.copy(bsz=B * T, device=tensor.device)
        flat = tensor.reshape(B * T, H)
        out = self.q_layer(flat, dev)
        return out.reshape(B, T, H)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = torch.stack([self._quantum_head(q[:, :, h, :]) for h in range(self.num_heads)], dim=2)
        k = torch.stack([self._quantum_head(k[:, :, h, :]) for h in range(self.num_heads)], dim=2)
        v = torch.stack([self._quantum_head(v[:, :, h, :]) for h in range(self.num_heads)], dim=2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realized by a quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._build_q_layer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, device='cpu')
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _build_q_layer(self, n_wires: int):
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, dev: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(dev, x)
                for w, gate in enumerate(self.params):
                    gate(dev, wires=w)
                return self.measure(dev)
        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n_qubits = self.q_layer.n_wires
        flat = x.reshape(B * T, C)
        if C < n_qubits:
            padded = torch.cat([flat, torch.zeros(B * T, n_qubits - C, device=flat.device)], dim=1)
        else:
            padded = flat[:, :n_qubits]
        dev = tq.QuantumDevice(n_wires=n_qubits, bsz=B * T, device=flat.device)
        q_out = self.q_layer(padded, dev)
        q_out = q_out.reshape(B, T, n_qubits)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Quantum transformer block with layer‑norm, quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_per_head: int = 2,
        n_qubits_ffn: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_qubits_per_head)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridTransformer(nn.Module):
    """Transformer‑based text classifier with quantum attention and feed‑forward."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        max_len: int = 512,
        n_qubits_per_head: int = 2,
        n_qubits_ffn: int = 4,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                n_qubits_per_head,
                n_qubits_ffn,
                dropout,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 1 else 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ['HybridTransformer']
