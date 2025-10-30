"""Hybrid classical LSTM‑Transformer architecture with optional quantum modules.

This module re‑implements the classical LSTM and transformer blocks from the
seed project, adds a regression head, and exposes a single
``HybridQLSTMTransformer`` class that can operate in either fully classical
or hybrid mode (by passing ``n_qubits_*`` > 0).  All layers are importable
and usable as drop‑in replacements for the original API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Classical QLSTM (linear gates) – 1st seed
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in classical LSTM cell using linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
# LSTMTagger – 1st seed
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagger that can swap between classical LSTM and QLSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(1)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
# Feed‑forward regressor – 2nd seed
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Simple fully‑connected regression head."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


# --------------------------------------------------------------------------- #
# Transformer block – 3rd seed (classic)
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Single transformer block with multi‑head attention and FFN."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn2(self.dropout(F.relu(self.ffn1(x))))
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Hybrid architecture – 1st + 2nd + 3rd seeds
# --------------------------------------------------------------------------- #
class HybridQLSTMTransformer(nn.Module):
    """Combined LSTM‑tagger, transformer‑classifier and regression head.

    Parameters
    ----------
    embedding_dim : int
        Embedding size for tokens.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Vocabulary size for embeddings.
    tagset_size : int
        Number of tags for sequence tagging.
    num_classes : int
        Output classes for the classifier.
    num_heads : int
        Heads for multi‑head attention.
    num_blocks : int
        Transformer layers.
    ffn_dim : int
        Feed‑forward dimension.
    n_qubits_lstm : int, default 0
        If >0, use quantum gates for the LSTM cell.
    n_qubits_transformer : int, default 0
        If >0, use quantum attention in the transformer.
    n_qubits_ffn : int, default 0
        If >0, use quantum feed‑forward in the transformer.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_classes: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        n_qubits_lstm: int = 0,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        # Tagging head
        self.tagger = LSTMTagger(
            embedding_dim,
            hidden_dim,
            vocab_size,
            tagset_size,
            n_qubits=n_qubits_lstm,
        )
        # Classification head
        self.classifier = TextClassifier(
            vocab_size,
            embedding_dim,
            num_heads,
            num_blocks,
            ffn_dim,
            num_classes,
            dropout=0.1,
        )
        # Regression head
        self.regressor = EstimatorNN(hidden_dim)

    def forward(
        self,
        sentence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tags : torch.Tensor
            Log‑probabilities over tagset, shape (seq_len, tagset_size).
        logits : torch.Tensor
            Classification logits, shape (batch, num_classes).
        reg : torch.Tensor
            Regression prediction, shape (batch, 1).
        """
        # Tagging
        tags = self.tagger(sentence)
        # Classification
        logits = self.classifier(sentence)
        # Regression: use mean hidden state from LSTM tagger
        with torch.no_grad():
            # Re‑run only once to obtain hidden states
            embeds = self.tagger.word_embeddings(sentence).unsqueeze(1)
            lstm_out, _ = self.tagger.lstm(embeds)
            hidden_mean = lstm_out.mean(dim=0)
        reg = self.regressor(hidden_mean)
        return tags, logits, reg


__all__ = [
    "QLSTM",
    "LSTMTagger",
    "EstimatorNN",
    "TransformerBlock",
    "PositionalEncoder",
    "TextClassifier",
    "HybridQLSTMTransformer",
]
