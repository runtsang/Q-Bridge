"""Hybrid classical sequence model combining transformer, LSTM and sampler.

The class is a drop‑in replacement for the original `TextClassifier`
in QTransformerTorch.py while adding optional LSTM and sampler blocks
drawn from QLSTM.py and SamplerQNN.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum‑friendly classes from the anchor transformer module
from QTransformerTorch import (
    MultiHeadAttentionQuantum,
    FeedForwardQuantum,
    TransformerBlockQuantum,
    TextClassifier,
    PositionalEncoder,
)

# Import the quantum‑enhanced LSTM and sampler
from QLSTM import QLSTM, LSTMTagger
from SamplerQNN import SamplerQNN


class HybridSeqModel(nn.Module):
    """Hybrid transformer / LSTM architecture with an optional quantum sampler.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of output classes (>=2 for softmax, 1 for binary).
    hidden_dim : int, optional
        Hidden size for the LSTM component.
    use_transformer : bool, default=True
        Whether to use the transformer encoder.
    use_lstm : bool, default=False
        Whether to add a quantum‑enhanced LSTM after the embedding.
    use_sampler : bool, default=False
        Whether to pass the encoder output through a sampler before classification.
    use_quantum : bool, default=False
        If True, use quantum attention/FFN blocks; otherwise classical.
    n_qubits : int, default=0
        Number of qubits for each quantum block (0 disables quantum).
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

        # Build transformer encoder
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

        # Optional quantum LSTM after embedding
        if self.use_lstm:
            self.lstm = QLSTM(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits,
            )

        # Optional sampler
        if self.use_sampler:
            self.sampler = SamplerQNN()

        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Token indices of shape (batch, seq_len) or (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes) or (batch, 1).
        """
        # Ensure shape is (batch, seq_len)
        if x.dim() == 2 and x.size(0) < x.size(1):
            x = x.t()

        # Embedding + positional encoding
        embeddings = self.token_embedding(x)
        embeddings = self.pos_encoder(embeddings)

        # Transformer encoder
        if self.use_transformer:
            embeddings = self.transformer(embeddings)

        # LSTM encoder (expects seq_len, batch, embed_dim)
        if self.use_lstm:
            seq_len, batch, _ = embeddings.size()
            lstm_input = embeddings.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
            lstm_output, _ = self.lstm(lstm_input)
            # Collapse to (batch, embed_dim)
            embeddings = lstm_output.mean(dim=1)

        # Sampler
        if self.use_sampler:
            embeddings = self.sampler(embeddings)

        # Output logits
        logits = self.classifier(embeddings)
        return logits

__all__ = ["HybridSeqModel"]
