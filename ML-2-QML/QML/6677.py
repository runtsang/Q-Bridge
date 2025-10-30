"""HybridSelfAttentionTransformer – quantum‑enhanced implementation.

This module implements the same public interface as the classical
variant but replaces the attention and feed‑forward sub‑modules with
parameterised quantum circuits.  The quantum circuits are built with
TorchQuantum and can be executed on simulators or real devices.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# ----------------------------------------------------------------------
#  Quantum sub‑modules
# ----------------------------------------------------------------------
class QuantumAttention(tq.QuantumModule):
    """
    Variational circuit that transforms each token vector into a
    quantum‑encoded representation and then computes a similarity
    matrix.  The circuit is lightweight: a single layer of
    parameterised RX gates followed by a CNOT ladder.
    """

    def __init__(self, n_wires: int, n_params: int) -> None:
        super().__init__()
        self.n_wires = n_wires

        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(n_params)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, d_k) to be encoded.
        q_device : tq.QuantumDevice
            Quantum device with at least `n_wires` qubits.

        Returns
        -------
        torch.Tensor
            Quantum‑encoded vectors of shape (batch, seq_len, n_wires).
        """
        self.encoder(q_device, x)
        for idx, gate in enumerate(self.params):
            gate(q_device, wires=[idx % self.n_wires])
        # entangle neighbouring qubits
        for i in range(self.n_wires - 1):
            tqf.cnot(q_device, wires=[i, i + 1])
        return self.measure(q_device)


class QuantumFeedForward(tq.QuantumModule):
    """
    Quantum feed‑forward network that maps an embedded token to a
    higher‑dimensional representation.  After the quantum circuit
    the output is passed through two classical linear layers.
    """

    def __init__(self, n_qubits: int, ffn_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical post‑processing
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim) to be encoded.
        q_device : tq.QuantumDevice
            Quantum device with at least `n_qubits` qubits.

        Returns
        -------
        torch.Tensor
            Classical tensor of shape (batch, seq_len, embed_dim).
        """
        self.encoder(q_device, x)
        for idx, gate in enumerate(self.params):
            gate(q_device, wires=[idx % self.n_qubits])
        out = self.measure(q_device)
        # Classical linear layers
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# ----------------------------------------------------------------------
#  Quantum‑enhanced transformer block
# ----------------------------------------------------------------------
class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that uses QuantumAttention and QuantumFeedForward.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attention: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = QuantumAttention(
            n_wires=n_qubits_attention, n_params=n_qubits_attention
        )
        self.ffn = QuantumFeedForward(
            n_qubits=n_qubits_ffn, ffn_dim=ffn_dim, embed_dim=embed_dim
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).
        mask : torch.Tensor, optional
            Boolean mask where True indicates padding.
        """
        batch, seq_len, _ = x.size()
        q_device = tq.QuantumDevice(n_wires=self.attn.n_wires, bsz=batch, device=x.device)

        # Encode queries, keys, values
        q_vec = self.attn(x, q_device)
        k_vec = self.attn(x, q_device)
        v_vec = self.attn(x, q_device)

        # Compute similarity scores
        scores = torch.einsum("bsi,bsj->bij", q_vec, k_vec) / math.sqrt(self.attn.n_wires)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bij,bjk->bik", attn_weights, v_vec)

        # Residual + norm
        x = self.norm1(x + self.dropout(attn_out))

        # Feed‑forward
        q_device_ffn = tq.QuantumDevice(n_wires=self.ffn.n_qubits, bsz=batch, device=x.device)
        ffn_out = self.ffn(x, q_device_ffn)

        # Residual + norm
        return self.norm2(x + self.dropout(ffn_out))


# ----------------------------------------------------------------------
#  Positional encoding (identical to classical variant)
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ----------------------------------------------------------------------
#  HybridSelfAttentionTransformer – quantum‑enhanced
# ----------------------------------------------------------------------
class HybridSelfAttentionTransformer(nn.Module):
    """
    Quantum‑enhanced transformer‑style classifier.
    The public API matches the classical implementation; the
    internal attention and feed‑forward layers are replaced by
    quantum modules.
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
        n_qubits_attention: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attention,
                    n_qubits_ffn,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = ["HybridSelfAttentionTransformer"]
