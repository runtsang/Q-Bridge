"""UnifiedHybridClassifier: classical image classifier and sequence tagger.

This module provides a single class that can be used either as a
convolutional binary classifier or as a sequence tagging model.
The implementation is fully classical and follows the structure of
the original seeds while adding a unified API that dispatches
between the two tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1.  Classical CNN classifier (drop‑in replacement of QCNet)
# --------------------------------------------------------------------------- #
class ClassicalCNNClassifier(nn.Module):
    """CNN‑based binary classifier with a sigmoid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = torch.sigmoid(x).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

# --------------------------------------------------------------------------- #
# 2.  Classical LSTM tagger (drop‑in for LSTMTagger)
# --------------------------------------------------------------------------- #
class ClassicalLSTMTagger(nn.Module):
    """Sequence tagging model that uses a standard nn.LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# 3.  UnifiedHybridClassifier
# --------------------------------------------------------------------------- #
class UnifiedHybridClassifier(nn.Module):
    """Unified model that can act as either a classifier or a tagger.

    Parameters
    ----------
    task : str
        One of ``'classification'`` or ``'tagging'``.  The default
        is ``'classification'``.
    kwargs : dict
        Additional keyword arguments are forwarded to the underlying
        sub‑module constructors.
    """
    def __init__(self, task: str = 'classification', **kwargs) -> None:
        super().__init__()
        if task not in {'classification', 'tagging'}:
            raise ValueError("task must be 'classification' or 'tagging'")
        self.task = task
        if task == 'classification':
            self.model = ClassicalCNNClassifier()
        else:
            # Expect kwargs: embedding_dim, hidden_dim, vocab_size, tagset_size
            self.model = ClassicalLSTMTagger(**kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Dispatches to the underlying sub‑module."""
        return self.model(*args, **kwargs)

__all__ = ["UnifiedHybridClassifier"]
