"""Hybrid classical model combining QCNN feature extraction with LSTM for sequence tagging."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical LSTM and QCNN from the seed modules
from.QLSTM import QLSTM as ClassicalQLSTM
from.QCNN import QCNNModel

class HybridQLSTMTagger(nn.Module):
    """
    A hybrid sequence tagging model that first applies a classical QCNN
    to embed each token, then feeds the resulting representations
    into a classical LSTM.
    """

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
        # Map embeddings to QCNN input size (8)
        self.feature_mapper = nn.Linear(embedding_dim, 8)
        # Classical QCNN extractor
        self.qcnn = QCNNModel()
        # Map QCNN output to hidden dimension
        self.qcnn_hidden_mapper = nn.Linear(1, hidden_dim)
        # Classical LSTM that processes the QCNN features
        self.lstm = ClassicalQLSTM(self.hidden_dim, self.hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.

        Args:
            sentence: Tensor of shape (seq_len, batch_size) containing word indices.

        Returns:
            Log‑softmaxed tag logits of shape (seq_len, batch_size, tagset_size).
        """
        # Embed words
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        seq_len, batch_size, _ = embeds.shape
        # Map embeddings to QCNN input size
        x = self.feature_mapper(embeds)  # (seq_len, batch, 8)
        # Flatten for QCNN
        x = x.reshape(seq_len * batch_size, 8)
        # QCNN forward
        x = self.qcnn(x)  # (seq_len*batch, 1)
        # Map QCNN output to hidden dimension
        x = self.qcnn_hidden_mapper(x)  # (seq_len*batch, hidden_dim)
        # Reshape back to sequence format
        x = x.reshape(seq_len, batch_size, self.hidden_dim)
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (seq_len, batch, hidden_dim)
        # Tag prediction
        tag_logits = self.hidden2tag(lstm_out.view(seq_len, -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTMTagger"]
