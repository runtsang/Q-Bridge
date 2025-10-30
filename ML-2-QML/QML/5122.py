import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Optional, List

class HybridFunction(nn.Module):
    """Differentiable sigmoid with optional shift, used as the quantum‑expectation head."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=8)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        q = self.separate_heads(x)
        k = self.separate_heads(x)
        v = self.separate_heads(x)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            self.q_layer(qdev)
            out = self.measure(qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, n_qubits: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QTransformerHybrid(nn.Module):
    """
    Hybrid transformer that can mix classical and quantum blocks.
    The quantum_blocks list indicates the indices of blocks that should use quantum
    attention/FFN.  Remaining blocks fall back to the classical implementation.
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
        quantum_blocks: Optional[List[int]] = None,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        quantum_blocks = quantum_blocks or []
        for i in range(num_blocks):
            if i in quantum_blocks:
                self.blocks.append(TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout, n_qubits))
            else:
                # Classical block
                self.blocks.append(TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout, n_qubits))  # placeholder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.head = HybridFunction(shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.token_emb(x)
        t = self.pos_enc(t)
        for blk in self.blocks:
            t = blk(t)
        t = self.dropout(t.mean(dim=1))
        logits = self.classifier(t)
        return self.head(logits)

    def evaluate(self, parameter_sets: List[List[float]], shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        """
        Fast evaluation with optional shot noise.  Works for both classical and quantum back‑ends.
        """
        self.eval()
        with torch.no_grad():
            raw = [self(torch.tensor(p).unsqueeze(0)).cpu().numpy().flatten().tolist() for p in parameter_sets]
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

__all__ = [
    "HybridFunction",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformerHybrid",
]
