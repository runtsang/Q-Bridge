"""
Hybrid classifier and LSTM module with a classical fallback.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridClassifierQLSTM", "build_classical_classifier",
           "build_classical_lstm", "ClassicalLSTMTagger"]


def build_classical_classifier(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a deep feed‑forward network that mirrors the parameter
    count of the quantum data‑uploading circuit described in the original seed.

    Returns:
        network: nn.Sequential containing all layers.
        encoding: list of indices that map each input feature to a single neuron.
        weight_sizes: list containing the weight‑size of each linear layer.
        observables: list of dummy observables to keep API compatibility.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class ClassicalLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses a standard nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int) -> None:
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


def build_classical_lstm(embedding_dim: int,
                         hidden_dim: int,
                         vocab_size: int,
                         tagset_size: int) -> nn.Module:
    """
    Factory that returns a ClassicalLSTMTagger instance.
    """
    return ClassicalLSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size)


class HybridClassifierQLSTM:
    """
    Classical fallback implementation of the hybrid classifier and LSTM.
    """
    def __init__(self,
                 num_features: int,
                 classifier_depth: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_quantum: bool = False) -> None:
        # Build classical components
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classical_classifier(
            num_features, classifier_depth
        )
        self.lstm = build_classical_lstm(embedding_dim, hidden_dim, vocab_size, tagset_size)
        self.use_quantum = use_quantum

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        """
        return self.classifier(x)

    def forward_lstm(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM tagger.
        """
        return self.lstm(sentence)
