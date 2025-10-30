import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1. Shared base classes (identical to the classical version)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Abstract base for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

class FeedForwardBase(nn.Module):
    """Abstract feed‑forward network base."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class TransformerBlockBase(nn.Module):
    """Base for a transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

# --------------------------------------------------------------------------- #
# 2. Classical implementations (for API symmetry)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using PyTorch's MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP with GELU activation."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlockClassical(TransformerBlockBase):
    """Purely classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 3. Quantum‑enhanced attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each head is processed by a 4‑qubit variational circuit."""
    class QLayer(tq.QuantumModule):
        """Quantum sub‑module that maps a d‑dim vector to a new d‑dim vector."""
        def __init__(self, d_k: int):
            super().__init__()
            self.d_k = d_k
            # Encode each input dimension as a rotation on a dedicated qubit
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
            )
            # Trainable RX gates
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(d_k)])
            # Entanglement pattern (ring)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for i in range(self.d_k - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.d_k - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits_per_head: int = 4, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_layer = self.QLayer(self.d_k)
        self.n_qubits_per_head = n_qubits_per_head

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        # Classical linear projections
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # Split heads
        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)
        # Quantum processing per head
        q_out = []
        k_out = []
        v_out = []
        for head in range(self.num_heads):
            # Flatten per head
            q_head = q[:, head, :, :].reshape(batch * seq_len, self.d_k)
            k_head = k[:, head, :, :].reshape(batch * seq_len, self.d_k)
            v_head = v[:, head, :, :].reshape(batch * seq_len, self.d_k)
            # Quantum device for this head
            qdev_q = tq.QuantumDevice(n_wires=self.d_k, bsz=batch * seq_len, device=q_head.device)
            qdev_k = tq.QuantumDevice(n_wires=self.d_k, bsz=batch * seq_len, device=k_head.device)
            qdev_v = tq.QuantumDevice(n_wires=self.d_k, bsz=batch * seq_len, device=v_head.device)
            # Apply quantum layer
            q_head_out = self.q_layer(q_head, qdev_q)
            k_head_out = self.q_layer(k_head, qdev_k)
            v_head_out = self.q_layer(v_head, qdev_v)
            # Reshape back
            q_out.append(q_head_out.reshape(batch, seq_len, self.d_k))
            k_out.append(k_head_out.reshape(batch, seq_len, self.d_k))
            v_out.append(v_head_out.reshape(batch, seq_len, self.d_k))
        # Stack heads
        q = torch.stack(q_out, dim=1)
        k = torch.stack(k_out, dim=1)
        v = torch.stack(v_out, dim=1)
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.combine_heads(attn_output)

# --------------------------------------------------------------------------- #
# 4. Quantum‑enhanced feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x_flat = x.view(batch * seq_len, self.q_layer.n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=batch * seq_len, device=x.device)
        q_out = self.q_layer(x_flat, qdev)
        q_out = q_out.view(batch, seq_len, self.q_layer.n_qubits)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
# 5. Hybrid feed‑forward (classical + quantum)
# --------------------------------------------------------------------------- #
class HybridFeedForward(nn.Module):
    """Weighted mix of classical and quantum feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 alpha: float = 0.5, dropout: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.classical = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.quantum = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.classical(x) + (1.0 - self.alpha) * self.quantum(x)

# --------------------------------------------------------------------------- #
# 6. Quantum‑enhanced transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and optionally hybrid feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int = 4, n_qubits_ffn: int = 4,
                 alpha_ffn: float = 0.5, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits_per_head=n_qubits_transformer)
        if alpha_ffn > 0.0:
            self.ffn = HybridFeedForward(embed_dim, ffn_dim, n_qubits_ffn,
                                         alpha=alpha_ffn, dropout=dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 7. Quantum‑adaptive positional encoder
# --------------------------------------------------------------------------- #
class PositionalEncoderQuantum(nn.Module):
    """Sinusoidal positional encoding augmented by a learnable quantum phase shift."""
    def __init__(self, embed_dim: int, max_len: int = 5000,
                 n_qubits_phase: int = 8):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        # Quantum phase module
        self.phase_q = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i % n_qubits_phase]} for i in range(embed_dim)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.n_qubits_phase = n_qubits_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x + self.pe[:, :x.size(1)]
        # Generate a phase shift using a quantum device
        batch, seq_len, embed_dim = x.shape
        qdev = tq.QuantumDevice(n_wires=self.n_qubits_phase, bsz=batch * seq_len, device=x.device)
        phase = self.phase_q(qdev, torch.zeros(batch * seq_len, self.n_qubits_phase))
        phase = phase.view(batch, seq_len, -1)  # shape (batch, seq_len, n_qubits_phase)
        # Broadcast to embed_dim (simple repeat)
        phase = phase.repeat(1, 1, embed_dim // self.n_qubits_phase + 1)[:, :, :embed_dim]
        return base + phase

# --------------------------------------------------------------------------- #
# 8. Quantum‑ready text classifier
# --------------------------------------------------------------------------- #
class TextClassifierQuantum(nn.Module):
    """Transformer‑based text classifier that can swap classical or quantum blocks."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        n_qubits_transformer: int = 4,
        n_qubits_ffn: int = 4,
        alpha_ffn: float = 0.5,
        pos_quantum: bool = False
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoderQuantum(embed_dim) if pos_quantum else PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if use_quantum:
                blocks.append(
                    TransformerBlockQuantum(
                        embed_dim, num_heads, ffn_dim,
                        n_qubits_transformer=n_qubits_transformer,
                        n_qubits_ffn=n_qubits_ffn,
                        alpha_ffn=alpha_ffn,
                        dropout=dropout
                    )
                )
            else:
                blocks.append(
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                )
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "HybridFeedForward",
    "PositionalEncoder",
    "PositionalEncoderQuantum",
    "TextClassifier",
    "TextClassifierQuantum",
]
