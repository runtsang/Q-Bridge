import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(2, 3)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class QLayer(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        self.measure(q_device)
        return q_device.get_expectation_value()


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires_per_head: Optional[int] = None, use_bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires_per_head = n_wires_per_head or self.d_k
        self.q_layer = QLayer(self.n_wires_per_head)
        self.out_proj = nn.Linear(self.n_wires_per_head, self.d_k, bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        flat = batch * seq_len * self.num_heads
        q_flat = q.reshape(flat, self.d_k)
        k_flat = k.reshape(flat, self.d_k)
        v_flat = v.reshape(flat, self.d_k)
        if self.n_wires_per_head!= self.d_k:
            self.map_to_wires = nn.Linear(self.d_k, self.n_wires_per_head, bias=False).to(x.device)
        else:
            self.map_to_wires = nn.Identity()
        q_wires = self.map_to_wires(q_flat)
        k_wires = self.map_to_wires(k_flat)
        v_wires = self.map_to_wires(v_flat)
        q_device = tq.QuantumDevice(n_wires=self.n_wires_per_head, bsz=flat, device=x.device)
        q_out = self.q_layer(q_wires, q_device)
        k_out = self.q_layer(k_wires, q_device)
        v_out = self.q_layer(v_wires, q_device)
        q_out = self.out_proj(q_out)
        k_out = self.out_proj(k_out)
        v_out = self.out_proj(v_out)
        q_out = q_out.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(2, 3)
        k_out = k_out.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(2, 3)
        v_out = v_out.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(2, 3)
        attn_output, _ = self.attention(q_out, k_out, v_out, mask)
        return attn_output.transpose(2, 3).contiguous().view(batch, seq_len, self.embed_dim)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int, dropout: float = 0.1,
                 use_bias: bool = True) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_wires = n_wires
        self.q_layer = QLayer(n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        flat = batch * seq_len
        x_flat = x.reshape(flat, -1)
        if self.n_wires!= self.embed_dim:
            self.map_to_wires = nn.Linear(self.embed_dim, self.n_wires, bias=False).to(x.device)
        else:
            self.map_to_wires = nn.Identity()
        x_wires = self.map_to_wires(x_flat)
        q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=flat, device=x.device)
        q_out = self.q_layer(x_wires, q_device)
        q_out = self.linear1(self.dropout(q_out))
        q_out = self.linear2(F.relu(q_out))
        return q_out.reshape(batch, seq_len, self.embed_dim)


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_wires_attention: Optional[int] = None, n_wires_ffn: Optional[int] = None,
                 dropout: float = 0.1, freeze_norm: bool = False,
                 residual_scale: float = 1.0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_wires_per_head=n_wires_attention)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                      n_wires=n_wires_ffn or embed_dim, dropout=dropout)
        if freeze_norm:
            for param in self.norm1.parameters():
                param.requires_grad = False
            for param in self.norm2.parameters():
                param.requires_grad = False
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out) * self.residual_scale)
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out) * self.residual_scale)


class PositionalEncoder(nn.Module):
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


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 n_wires_attention: Optional[int] = None, n_wires_ffn: Optional[int] = None,
                 freeze_norm: bool = False, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                      n_wires_attention=n_wires_attention,
                                      n_wires_ffn=n_wires_ffn, dropout=dropout,
                                      freeze_norm=freeze_norm,
                                      residual_scale=residual_scale)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["MultiHeadAttentionBase", "MultiHeadAttentionQuantum",
           "FeedForwardBase", "FeedForwardQuantum",
           "TransformerBlockBase", "TransformerBlockQuantum",
           "PositionalEncoder", "TextClassifier"]
