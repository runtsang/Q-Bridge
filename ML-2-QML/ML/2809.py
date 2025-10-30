"""QuantumHybridNAT: Classical-only implementation for image‑to‑sequence tasks.

The class mirrors the quantum‑enhanced version but uses fully classical
components.  It can be used as a drop‑in replacement for the original
Quantum‑NAT model in any training pipeline that expects a purely
classical PyTorch module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumHybridNAT"]


class _CNNBackbone(nn.Module):
    """CNN backbone identical to the original Quantum‑NAT model."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _ImageEncoder(nn.Module):
    """Classical image encoder: CNN + fully‑connected head."""

    def __init__(self, cnn_channels: list[int], fc_hidden: int) -> None:
        super().__init__()
        self.cnn = _CNNBackbone(cnn_channels[0], cnn_channels[1:3])
        # 16 feature maps, 7x7 spatial size after two 2x2 pools
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out)


class _SeqEncoder(nn.Module):
    """Classical sequence encoder that uses an LSTM."""

    def __init__(self, vocab_size: int, embedding_dim: int, lstm_hidden: int, tagset_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=False)
        self.hidden2tag = nn.Linear(lstm_hidden, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: (B, T)
        embeds = self.embedding(sentence)  # (B, T, embed_dim)
        embeds = embeds.permute(1, 0, 2)  # (T, B, embed_dim)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=2)


class QuantumHybridNAT(nn.Module):
    """Hybrid model that combines a CNN image encoder with a sequence tagger.

    Parameters
    ----------
    n_wires : int
        Number of quantum wires used in the quantum variant (ignored here).
    qlstm_qubits : int
        Number of qubits for the quantum LSTM (ignored here).
    cnn_channels : list[int]
        Channel configuration for the CNN backbone.
    fc_hidden : int
        Size of the hidden layer in the fully‑connected head.
    lstm_hidden : int
        Hidden size of the LSTM sequence encoder.
    vocab_size : int
        Vocabulary size for the embedding layer.
    embedding_dim : int
        Dimension of the word embeddings.
    tagset_size : int
        Number of output tags.
    """

    def __init__(
        self,
        n_wires: int,
        qlstm_qubits: int,
        cnn_channels: list[int],
        fc_hidden: int,
        lstm_hidden: int,
        vocab_size: int,
        embedding_dim: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.image_encoder = _ImageEncoder(cnn_channels, fc_hidden)
        self.seq_encoder = _SeqEncoder(vocab_size, embedding_dim, lstm_hidden, tagset_size)

    def forward(
        self,
        image: torch.Tensor,
        sentence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image features and sequence tag logits."""
        img_out = self.image_encoder(image)
        seq_out = self.seq_encoder(sentence)
        return img_out, seq_out
