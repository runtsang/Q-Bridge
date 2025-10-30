"""HybridNAT: purely classical implementation with optional classification or tagging modes.

The class mirrors the architecture of the original QuantumNAT seed, but
removes all quantum dependencies.  It retains a CNN backbone for image
classification and a classical LSTM for sequence tagging.  This makes the
model a drop‑in replacement when a quantum backend is not available.

The API is intentionally minimal: construct with ``mode="classification"``
or ``mode="tagging"`` and call ``forward`` on an appropriate input tensor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridNAT"]


class HybridNAT(nn.Module):
    """
    Classical hybrid model with two operating modes.

    Parameters
    ----------
    mode : {"classification", "tagging"}
        Select the task.  ``classification`` uses a CNN + FC head,
        ``tagging`` uses an embedding + LSTM + linear tagger.
    in_channels : int, default 1
        Number of input channels for the CNN.
    num_classes : int, default 4
        Number of classes for classification.
    hidden_dim : int, default 64
        Hidden dimension for the FC head and the LSTM.
    embedding_dim : int, default 128
        Embedding size for the LSTM tagger.
    vocab_size : int, default 5000
        Vocabulary size for embeddings.
    tagset_size : int, default 10
        Number of tags for sequence tagging.
    """

    def __init__(
        self,
        mode: str = "classification",
        *,
        in_channels: int = 1,
        num_classes: int = 4,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        vocab_size: int = 5000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "classification":
            # Feature extractor
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Compute flattened feature size
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, 28, 28)
                feats = self.features(dummy)
                flattened_size = feats.view(1, -1).size(1)
            self.fc = nn.Sequential(
                nn.Linear(flattened_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
            self.norm = nn.BatchNorm1d(num_classes)
        elif self.mode == "tagging":
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "classification":
            feats = self.features(x)
            flattened = feats.view(feats.size(0), -1)
            out = self.fc(flattened)
            return self.norm(out)
        else:  # tagging
            embeds = self.embedding(x)  # shape: (seq_len, batch_size, embed_dim)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)
