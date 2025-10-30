"""Hybrid LSTM tagging model with optional classical classifier.

This module provides a drop‑in replacement for the original QLSTM
implementations, but with a fully classical backbone. It exposes
`QLSTMClassifier`, a single class that can be instantiated in either
classical or quantum mode (the quantum mode is handled by the QML
module).  The classical branch uses PyTorch’s `nn.LSTM` and a simple
feed‑forward classifier built by :func:`build_classifier_circuit`.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a simple feed‑forward classifier that mimics the interface
    of the quantum circuit builder.  The returned tuple contains:

    1. a :class:`torch.nn.Sequential` network,
    2. a list of input feature indices (identity mapping),
    3. a list of the number of trainable parameters per layer,
    4. a list of “observable” indices (here simply the output classes).

    Parameters
    ----------
    num_features : int
        Dimensionality of the input features.
    depth : int
        Number of hidden layers.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding: list[int] = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class QLSTMClassifier(nn.Module):
    """
    Hybrid sequence‑tagging model that can operate in classical mode
    (default) or delegate to a quantum backend via the QML module.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Hidden state dimensionality of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of distinct tags.
    n_qubits : int, optional
        If greater than zero, the quantum backend is used.  In the
        classical implementation this flag is ignored.
    classifier_depth : int, optional
        Depth of the feed‑forward classifier.
    classifier_n_qubits : int, optional
        Number of qubits for the quantum classifier head.  Ignored in
        the classical implementation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        classifier_depth: int = 2,
        classifier_n_qubits: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Classical LSTM backbone
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Classifier head – classical feed‑forward network
        self.classifier_network, _, _, _ = build_classifier_circuit(
            num_features=hidden_dim, depth=classifier_depth
        )

        # Tag projection
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices with shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Log‑probabilities over tags for each token.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMClassifier", "build_classifier_circuit"]
