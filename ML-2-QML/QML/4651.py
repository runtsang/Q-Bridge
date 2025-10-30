"""Quantum‑enhanced hybrid sequence model.

This module mirrors the classical `HybridSeqModel` but replaces
transformer attention and FFN blocks with their quantum counterparts
and employs a quantum LSTM from QLSTM.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Import quantum‑enhanced classes
from QTransformerTorch import (
    MultiHeadAttentionQuantum,
    FeedForwardQuantum,
    TransformerBlockQuantum,
    PositionalEncoder,
)
from QLSTM import QLSTM
from SamplerQNN import SamplerQNN


class HybridSeqModelQuantum(nn.Module):
    """Quantum‑enabled hybrid transformer / LSTM architecture with sampler.

    Parameters mirror the classical version; quantum blocks are activated when
    `use_quantum=True` and `n_qubits>0`.  The sampler is a Qiskit
    parameterised circuit that can be attached before the final classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        *,
        use_transformer: bool = True,
        use_lstm: bool = False,
        use_sampler: bool = False,
        use_quantum: bool = False,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()

        self.use_transformer = use_transformer
        self.use_lstm = use_lstm
        self.use_sampler = use_sampler

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Build transformer encoder with optional quantum blocks
        if self.use_transformer:
            if use_quantum and n_qubits > 0:
                self.transformer = nn.Sequential(
                    *[
                        TransformerBlockQuantum(
                            embed_dim,
                            num_heads,
                            ffn_dim,
                            n_qubits_transformer=n_qubits,
                            n_qubits_ffn=n_qubits,
                            n_qlayers=1,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            else:
                self.transformer = nn.Sequential(
                    *[
                        TransformerBlockQuantum(
                            embed_dim,
                            num_heads,
                            ffn_dim,
                            n_qubits_transformer=0,
                            n_qubits_ffn=0,
                            n_qlayers=1,
                        )
                        for _ in range(num_blocks)
                    ]
                )

        # Quantum LSTM
        if self.use_lstm:
            self.lstm = QLSTM(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits,
            )

        # Quantum sampler
        if self.use_sampler:
            self.sampler = SamplerQNN()

        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantum model.

        Parameters
        ----------
        x : torch.Tensor
            Token indices of shape (batch, seq_len) or (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes) or (batch, 1).
        """
        if x.dim() == 2 and x.size(0) < x.size(1):
            x = x.t()

        embeddings = self.token_embedding(x)
        embeddings = self.pos_encoder(embeddings)

        if self.use_transformer:
            embeddings = self.transformer(embeddings)

        if self.use_lstm:
            seq_len, batch, _ = embeddings.size()
            lstm_input = embeddings.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
            lstm_output, _ = self.lstm(lstm_input)
            embeddings = lstm_output.mean(dim=1)

        if self.use_sampler:
            embeddings = self.sampler(embeddings)

        logits = self.classifier(embeddings)
        return logits

__all__ = ["HybridSeqModelQuantum"]
