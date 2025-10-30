"""
QuantumTransformerWrapper: Quantum‑enhanced transformer with variational circuits.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
# 1. Base classes – identical to seed
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward variants."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


# --------------------------------------------------------------------------- #
# 2. Classical implementations – identical to seed
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        # split heads
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine_heads(attn)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Pure classical transformer block."""
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
# 3. Quantum implementations – variational circuits
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that uses a variational quantum circuit to generate Q, K, V."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qubits: int = 0,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive for quantum attention")
        if n_qubits % num_heads!= 0:
            raise ValueError("n_qubits must be divisible by num_heads")
        self.n_qubits = n_qubits
        self.d_k = n_qubits // num_heads
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        # encoder: each input dimension -> RX on corresponding wire
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.combine_heads = nn.Linear(n_qubits, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _quantum_forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        """Run the variational circuit on a single token embedding."""
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        # slice first n_qubits dimensions to feed into the quantum circuit
        x_slice = x[..., : self.n_qubits]
        # run quantum circuit for each token
        q_vecs = torch.empty(batch, seq, self.n_qubits, device=x.device, dtype=x.dtype)
        for b in range(batch):
            for s in range(seq):
                token = x_slice[b, s]
                qdev = self.q_device.copy(bsz=1, device=token.device)
                q_vecs[b, s] = self._quantum_forward(token, qdev)
        # treat quantum vectors as Q, K, V
        q = q_vecs.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = q.clone()
        v = q.clone()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.n_qubits)
        attn = self.combine_heads(attn)
        return attn


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a variational quantum circuit."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        if n_qubits_ffn <= 0:
            raise ValueError("n_qubits_ffn must be positive for quantum feed‑forward")
        self.n_qubits_ffn = n_qubits_ffn
        self.q_device = tq.QuantumDevice(n_wires=n_qubits_ffn)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits_ffn)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits_ffn)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits_ffn, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def _quantum_forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x_slice = x[..., : self.n_qubits_ffn]
        q_vecs = torch.empty(batch, seq, self.n_qubits_ffn, device=x.device, dtype=x.dtype)
        for b in range(batch):
            for s in range(seq):
                token = x_slice[b, s]
                qdev = self.q_device.copy(bsz=1, device=token.device)
                q_vecs[b, s] = self._quantum_forward(token, qdev)
        out = self.linear1(q_vecs)
        out = self.linear2(self.dropout(F.relu(out)))
        return out


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_qlayers: int,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, n_qubits_transformer, q_device
        )
        # choose quantum or classical feed‑forward depending on n_qubits_ffn
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 4. Positional encoding – unchanged
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 5. Quantum feature extractor – lightweight quantum module
# --------------------------------------------------------------------------- #
class QuantumFeatureExtractor(nn.Module):
    """Extracts a quantum feature vector from token embeddings."""
    def __init__(self, n_qubits: int, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive for quantum feature extraction")
        self.n_qubits = n_qubits
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _quantum_forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq, embed_dim).
        Returns
        -------
        torch.Tensor
            Quantum feature tensor of shape (batch, seq, n_qubits).
        """
        batch, seq, _ = x.shape
        x_slice = x[..., : self.n_qubits]
        out = torch.empty(batch, seq, self.n_qubits, device=x.device, dtype=x.dtype)
        for b in range(batch):
            for s in range(seq):
                token = x_slice[b, s]
                qdev = self.q_device.copy(bsz=1, device=token.device)
                out[b, s] = self._quantum_forward(token, qdev)
        return out


# --------------------------------------------------------------------------- #
# 6. QuantumTransformerWrapper – full quantum implementation
# --------------------------------------------------------------------------- #
class QuantumTransformerWrapper(nn.Module):
    """
    Wrapper that supports both classical and quantum transformer blocks.
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Dropout probability.
    use_quantum : bool, optional
        If True, quantum blocks are used (requires n_qubits_* > 0).
    n_qubits_transformer : int, optional
        Number of qubits for transformer‑level quantum layers.
    n_qubits_ffn : int, optional
        Number of qubits for feed‑forward quantum layers.
    n_qlayers : int, optional
        Number of variational layers (currently unused but kept for API).
    q_device : Any, optional
        Quantum device (used only when use_quantum is True).
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
        use_quantum: bool = False,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        if use_quantum and n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        n_qlayers,
                        q_device=q_device,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.quantum_extractor = QuantumFeatureExtractor(n_qubits_transformer, q_device=q_device)
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
            self.quantum_extractor = None

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns
        -------
        logits : torch.Tensor
            Classification logits of shape (batch, num_classes).
        quantum_features : torch.Tensor or None
            If quantum_extractor is present, a tensor of shape
            (batch, n_qubits_transformer) containing averaged quantum features.
            Otherwise None.
        """
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)

        if self.quantum_extractor is not None:
            # obtain quantum features from the final transformer output
            q_feats = self.quantum_extractor(x)  # shape (batch, seq, n_qubits)
            q_feats = q_feats.mean(dim=1)  # average over sequence
            return logits, q_feats
        else:
            return logits, None


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
    "PositionalEncoder",
    "QuantumFeatureExtractor",
    "QuantumTransformerWrapper",
]
