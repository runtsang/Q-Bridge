from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from.QLSTM import QLSTM as ClassicalQLSTM
from.Autoencoder import Autoencoder
from.QuantumClassifierModel import build_classifier_circuit

class HybridQLSTMClassifier(nn.Module):
    """
    Hybrid LSTM-based sequence tagger that optionally compresses embeddings
    with a classical autoencoder and uses a classical or quantum classifier head.
    The LSTM layer can be a standard nn.LSTM or the classical QLSTM gate
    implementation from the anchor module.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        classifier_depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Autoencoder for dimensionality reduction of embeddings
        self.autoencoder = Autoencoder(
            input_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Choose LSTM implementation
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(latent_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(latent_dim, hidden_dim)

        # Classifier head – either classical or quantum
        classifier_net, _, _, _ = build_classifier_circuit(
            num_features=hidden_dim, depth=classifier_depth
        )
        self.classifier = classifier_net
        self.output_layer = nn.Linear(2, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.
        :param sentence: Tensor of token indices, shape (seq_len, batch)
        :return: Log-probabilities over tags, shape (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        flat_embeds = embeds.view(-1, embeds.size(-1))
        latent = self.autoencoder.encode(flat_embeds)  # (seq_len*batch, latent_dim)
        latent_seq = latent.view(embeds.size(0), embeds.size(1), -1)

        lstm_out, _ = self.lstm(latent_seq)
        flat_lstm = lstm_out.view(-1, self.hidden_dim)
        logits = self.classifier(flat_lstm)  # (seq_len*batch, 2)
        logits = self.output_layer(logits)   # (seq_len*batch, tagset_size)
        logits = logits.view(embeds.size(0), embeds.size(1), -1)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTMClassifier"]
