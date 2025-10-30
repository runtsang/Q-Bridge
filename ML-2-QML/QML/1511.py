import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention modules."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.head_dim * self.num_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class QLayer(tq.QuantumModule):
    """Quantum module used in the multi‑head attention."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate, wire in zip(self.parameters, range(self.n_wires)):
            gate(qdev, wires=wire)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        return self.measure(qdev)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that applies quantum transformations to projections."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.head_dim)
        self.q_layers = nn.ModuleList([QLayer(self.head_dim) for _ in range(num_heads)])
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        batch, seq, _ = x.size()
        qdev = self.q_device.copy(bsz=batch, device=x.device)
        x_flat = x.view(batch * seq, self.head_dim)
        out = self.q_layers[head_idx](x_flat, qdev)
        return out.view(batch, seq, self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        q_quant = torch.stack([self._apply_quantum(q[:, h, :, :], h) for h in range(self.num_heads)], dim=1)
        k_quant = torch.stack([self._apply_quantum(k[:, h, :, :], h) for h in range(self.num_heads)], dim=1)
        v_quant = torch.stack([self._apply_quantum(v[:, h, :, :], h) for h in range(self.num_heads)], dim=1)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_quant, k_quant) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_probs, v_quant)
        attn_output = self.combine_heads(attn_output)
        return self.out_proj(attn_output)


class QFeedForwardLayer(tq.QuantumModule):
    """Quantum module for the feed‑forward network."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate, wire in zip(self.parameters, range(self.n_wires)):
            gate(qdev, wires=wire)
        return self.measure(qdev)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a quantum module."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = QFeedForwardLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        outputs = []
        for i in range(seq):
            token = x[:, i, :].clone()
            if token.size(1) > self.n_qubits:
                token = token[:, :self.n_qubits]
            qdev = self.q_device.copy(bsz=batch, device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑augmented transformer block."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridTransformer(nn.Module):
    """Transformer‑based text classifier with quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_pos_bias: bool = False,
        pos_bias_init: float = 0.0,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        gradient_sharpening: bool = False,
        grad_sharpen_thresh: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.use_pos_bias = use_pos_bias
        if use_pos_bias:
            self.pos_bias = nn.Parameter(torch.full((embed_dim,), pos_bias_init))
        else:
            self.pos_bias = None
        if n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        if gradient_sharpening:
            self._register_gradient_sharpening(grad_sharpen_thresh)

    def _register_gradient_sharpening(self, thresh: float) -> None:
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad, t=thresh: grad * (t / grad.norm()) if grad.norm() > t else grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        if self.pos_bias is not None:
            tokens = tokens + self.pos_bias
        tokens = self.pos_encoder(tokens)
        x = self.transformers(tokens)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformer",
]
