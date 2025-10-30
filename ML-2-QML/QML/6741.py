# QuantumConvTransformer – Quantum implementation using a 2×2 quantum convolution followed by a stack of quantum transformer blocks.

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumConvFilter:
    'Quantum filter that maps each 2×2 image patch to a scalar.'

    def __init__(self, n_qubits: int = 4, threshold: float = 0.0) -> None:
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def run(self, data, q_device: tq.QuantumDevice | None = None) -> float:
        'Execute the circuit on a 2×2 patch and return the average probability of |1>.'
        if isinstance(data, np.ndarray):
            data = torch.tensor(data.flatten(), dtype=torch.float32)
        qdev = q_device or tq.QuantumDevice(n_wires=self.n_qubits, bsz=1)
        self.encoder(qdev, data)
        for gate in self.params:
            gate(qdev)
        result = self.measure(qdev)
        return result.mean().item()

class MultiHeadAttentionBase(nn.Module):
    'Base class for attention layers.'

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   batch_size: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    'Standard multi‑head attention (used as a fallback).'

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, x.size(0), mask)
        return self.combine(out)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    'Attention where the linear projections are replaced by quantum circuits.'

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.params:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer(embed_dim)
        self.q_device = q_device or tq.QuantumDevice(n_wires=embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.unsqueeze(0)  # (1, embed_dim)
            qdev = self.q_device.copy(bsz=1, device=token.device)
            proj = self.q_layer(token, qdev)
            projections.append(proj)
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)
        out = self.downstream(q, k, v, x.size(0), mask)
        return self.combine(out)

class FeedForwardBase(nn.Module):
    'Base class for the feed‑forward network.'

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    'Classical two‑layer MLP.'

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    'Feed‑forward implemented by a quantum circuit.'

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.params:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    'Base transformer block.'

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    'Standard transformer block using classical attention and feed‑forward.'

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    'Transformer block that can use quantum attention and/or quantum feed‑forward.'

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int,
                 q_device: tq.QuantumDevice | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
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
    'Sinusoidal positional encoding.'

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

class QuantumConvTransformer(nn.Module):
    'Hybrid transformer with a quantum 2×2 convolution followed by quantum transformer blocks.'

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.qconv = QuantumConvFilter(n_qubits=4, threshold=0.0)
        self.proj = nn.Linear(1, embed_dim, bias=False)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        q_device=q_device,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            # fallback to classical blocks
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(
                        embed_dim, num_heads, ffn_dim, dropout
                    )
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, height, width), interpreted as a 2‑D image.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        batch_size = x.size(0)
        conv_outputs = []
        for i in range(batch_size):
            patch = x[i, :self.qconv.n_qubits // 2, :self.qconv.n_qubits // 2].cpu().numpy()
            conv_outputs.append(self.qconv.run(patch))
        conv_vec = torch.tensor(conv_outputs, device=x.device, dtype=torch.float32).unsqueeze(-1)
        projected = self.proj(conv_vec)  # (batch, 1, embed_dim)
        seq_len = x.size(1)  # reuse height as sequence length
        tokens = projected.repeat(1, seq_len, 1)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
