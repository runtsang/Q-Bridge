"""Classical implementation of the GraphQNNHybrid interface.

This module consolidates the graph utilities, LSTM tagger, quanvolution and QFCModel
from the four reference pairs into a single importable class.  Each sub‑module
mirrors the quantum counterpart but operates purely on CPU tensors.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphQNNHybrid:
    """Hybrid graph neural network utilities with classical implementation."""

    # --------------------------------------------------------------------- #
    # Graph utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_linear(in_features: int, out_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights = [GraphQNNHybrid.random_linear(in_f, out_f)
                   for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
        training_data = GraphQNNHybrid.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        stored = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # Classical LSTM tagger
    # --------------------------------------------------------------------- #
    class LSTMTagger(nn.Module):
        """Sequence tagging with a standard PyTorch LSTM."""

        def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            tagset_size: int,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence: torch.Tensor) -> torch.Tensor:
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds)
            tag_logits = self.hidden2tag(lstm_out)
            return F.log_softmax(tag_logits, dim=-1)

    # --------------------------------------------------------------------- #
    # Classical quanvolution
    # --------------------------------------------------------------------- #
    class QuanvolutionFilter(nn.Module):
        """2‑D convolution that emulates a quantum kernel on image patches."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.conv(x)
            return features.view(x.size(0), -1)

    class QuanvolutionClassifier(nn.Module):
        """Hybrid network: quanvolution front‑end + linear head."""

        def __init__(self) -> None:
            super().__init__()
            self.qfilter = GraphQNNHybrid.QuanvolutionFilter()
            self.linear = nn.Linear(4 * 14 * 14, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.qfilter(x)
            logits = self.linear(features)
            return F.log_softmax(logits, dim=-1)

    # --------------------------------------------------------------------- #
    # Classical QFCModel
    # --------------------------------------------------------------------- #
    class QFCModel(nn.Module):
        """Simple CNN followed by a fully‑connected projection to four features."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            features = self.features(x)
            flattened = features.view(bsz, -1)
            out = self.fc(flattened)
            return self.norm(out)


__all__ = ["GraphQNNHybrid"]
