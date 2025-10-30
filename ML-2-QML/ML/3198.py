"""Importable classical module defining UnifiedQuantumHybridLayer.

This module implements a hybrid architecture that can be used in a PyTorch
pipeline.  The class behaves like the original classical FCL+LSTMTagger
but is packaged as a single module for easy import.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical fully‑connected layer
class FCL(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas is a 1‑D tensor
        return torch.tanh(self.linear(thetas))

# Classical LSTM cell
class QLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, (hn, cn) = self.lstm(inputs)
        return output, (hn, cn)

# Sequence tagging model
class LSTMTagger(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

# Unified hybrid layer
class UnifiedQuantumHybridLayer(nn.Module):
    """
    Combines a fully‑connected layer with an LSTM for sequence tagging.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Hidden size of LSTM.
    vocab_size : int
        Size of vocabulary.
    tagset_size : int
        Number of output tags.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.tagger = LSTMTagger(
            embedding_dim, hidden_dim, vocab_size, tagset_size
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that delegates to the classical LSTMTagger.
        """
        return self.tagger(sentence)

__all__ = ["UnifiedQuantumHybridLayer"]
