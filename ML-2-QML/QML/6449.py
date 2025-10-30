"""QTransformer quantum implementation using TorchQuantum."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QLayer(tq.QuantumModule):
    """Quantum sub‑module used for attention and feed‑forward transformations."""
    def __init__(self, n_wires: int, n_ops: int = 20):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, vector: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, vector)
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        return self.measure(qdev)


class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where each head is processed by a quantum layer."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 4):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim!= n_wires:
            raise ValueError(f"head_dim ({self.head_dim}) must equal n_wires ({n_wires}) for quantum encoding")
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.q_layer = QLayer(n_wires, n_ops=20)

    def _apply_quantum(self, tensor: torch.Tensor) -> torch.Tensor:
        B, T, H, D = tensor.shape
        outputs = []
        for b in range(B):
            batch_out = []
            for h in range(H):
                head = tensor[b, :, h, :]  # (T, D)
                qdev = tq.QuantumDevice(n_wires=D, bsz=T, device=head.device)
                out = self.q_layer(qdev, head)
                batch_out.append(out)
            outputs.append(torch.stack(batch_out, dim=1))
        return torch.stack(outputs, dim=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_quantum(q)
        k = self._apply_quantum(k)
        v = self._apply_quantum(v)
        scores = torch.einsum("bthd,bshd->bths", q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1).unsqueeze(-1) == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.einsum("bths,bshd->bthd", attn_weights, v)
        out = out.reshape(B, T, C)
        return self.out_proj(out)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward block that uses a quantum layer before a linear projection."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_wires = n_wires
        self.q_layer = QLayer(n_wires, n_ops=20)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        outputs = []
        for b in range(B):
            batch_out = []
            for t in range(T):
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=x.device)
                out = self.q_layer(qdev, x[b, t, :].unsqueeze(0))
                batch_out.append(out)
            outputs.append(torch.stack(batch_out, dim=0))
        out = torch.stack(outputs, dim=0)
        out = self.linear1(out)
        out = self.dropout(out)
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Quantum transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, n_wires: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to the classical version)."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class QTransformer(nn.Module):
    """Transformer that can be instantiated in quantum mode."""
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_wires: int = 4,
    ):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads)!= n_wires:
            raise ValueError("head_dim must equal n_wires for quantum encoding")
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout, n_wires)
             for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes if num_classes > 1 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_dim] – input may be real or complex."""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.dropout(x.mean(dim=1))
        return self.head(x)


__all__ = [
    "QLayer",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformer",
]
