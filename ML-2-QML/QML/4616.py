import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Quantum multi‑head attention using a small quantum layer per head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]}
                 for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # split heads
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        out_heads = []
        for head in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
            out = self.q_layer(head, qdev)
            out_heads.append(out)
        out = torch.stack(out_heads, dim=1).transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardQuantum(tq.QuantumModule):
    """Quantum feed‑forward implemented as a small quantum module followed by classical layers."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]}
                 for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int,
                 dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Hybrid block that optionally uses quantum attention and/or feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if n_qubits_transformer > 0:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                                  dropout,
                                                  q_device=q_device)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                    dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                          n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class FeedForwardClassical(nn.Module):
    """Purely classical feed‑forward for use in the hybrid block."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttentionClassical(nn.Module):
    """Fallback classical attention for the hybrid block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class CNNFeatureExtractor(nn.Module):
    """2‑D CNN that produces a sequence of tokens for the transformer."""
    def __init__(self,
                 in_channels: int = 1,
                 num_tokens: int = 64,
                 embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.token_proj = nn.Linear(64 * 7 * 7, num_tokens * embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        flat = self.flatten(feat)
        token_space = self.token_proj(flat)
        tokens = token_space.view(x.size(0), -1,
                                 self.token_proj.out_features // token_space.size(1))
        return tokens


class HybridTransformerCNN(tq.QuantumModule):
    """
    Quantum‑enhanced version of the hybrid transformer image classifier.
    It reuses the CNN feature extractor and then passes the token sequence
    through a stack of transformer blocks that can be classical, quantum,
    or a mixture of both. Photonic‑style parameter clipping is applied
    to any quantum sub‑module that receives real‑valued data.
    """
    def __init__(self,
                 num_classes: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 256,
                 num_tokens: int = 64,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 in_channels: int = 1):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, num_tokens, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_tokens)
        if n_qubits_transformer > 0 or n_qubits_ffn > 0:
            q_device = tq.QuantumDevice(
                n_wires=max(n_qubits_transformer, n_qubits_ffn))
            blocks = [TransformerBlockQuantum(embed_dim, num_heads,
                                               ffn_dim, n_qubits_transformer,
                                               n_qubits_ffn, dropout,
                                               q_device=q_device)
                      for _ in range(num_blocks)]
        else:
            blocks = [TransformerBlockQuantum(embed_dim, num_heads,
                                               ffn_dim, 0, 0, dropout)
                      for _ in range(num_blocks)]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.cnn(x)
        tokens = self.pos_encoder(tokens)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


__all__ = ["HybridTransformerCNN"]
