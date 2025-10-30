"""
Classical implementation of a hybrid graph‑QNN / LSTM / classifier
framework.  All components are pure PyTorch / NumPy / networkx.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Graph‑QNN helpers
# --------------------------------------------------------------------------- #
Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialised weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight list, training data and target weight."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of activations for each sample."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  LSTM helpers
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Pure‑PyTorch LSTM cell that matches the quantum interface."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that can operate with a classical LSTM."""

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
            # In the classical module we still use a pure LSTM; the quantum
            # variant is in the quantum module.
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
#  Quantum classifier helper
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier mirroring the quantum version."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
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


# --------------------------------------------------------------------------- #
#  Unified class
# --------------------------------------------------------------------------- #
class GraphQNNGen:
    """
    A hybrid framework that supports three operational modes:

    * ``mode="graph"`` – classical feed‑forward graph QNN
    * ``mode="lstm"``  – sequence tagging with a classical LSTM
    * ``mode="classifier"`` – a simple classifier with a classical feed‑forward network

    The constructor accepts the same keyword arguments used in the
    original reference modules, but defaults are provided for quick
    experimentation.
    """

    def __init__(self, mode: str = "graph", **kwargs):
        self.mode = mode.lower()
        if self.mode == "graph":
            self.qnn_arch: Sequence[int] = kwargs.get("qnn_arch", [2, 4, 2])
            self.samples: int = kwargs.get("samples", 100)
            self.arch, self.weights, self.training_data, self.target = random_network(
                self.qnn_arch, self.samples
            )
        elif self.mode == "lstm":
            self.embedding_dim: int = kwargs.get("embedding_dim", 10)
            self.hidden_dim: int = kwargs.get("hidden_dim", 20)
            self.vocab_size: int = kwargs.get("vocab_size", 100)
            self.tagset_size: int = kwargs.get("tagset_size", 5)
            self.n_qubits: int = kwargs.get("n_qubits", 0)
            self.lstm_tagger = LSTMTagger(
                self.embedding_dim,
                self.hidden_dim,
                self.vocab_size,
                self.tagset_size,
                self.n_qubits,
            )
        elif self.mode == "classifier":
            self.num_features: int = kwargs.get("num_features", 10)
            self.depth: int = kwargs.get("depth", 3)
            self.classifier, _, _, _ = build_classifier_circuit(
                self.num_features, self.depth
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------- Graph QNN methods --------------------- #
    def graph_feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        if self.mode!= "graph":
            raise RuntimeError("graph_feedforward is only available in graph mode")
        return feedforward(self.arch, self.weights, samples)

    def graph_fidelity_graph(
        self, threshold: float, *, secondary: float | None = None
    ) -> nx.Graph:
        if self.mode!= "graph":
            raise RuntimeError("graph_fidelity_graph is only available in graph mode")
        activations = self.graph_feedforward(self.training_data)
        # Use the last layer activations as states
        states = [a[-1] for a in activations]
        return fidelity_adjacency(states, threshold, secondary=secondary)

    # --------------------- LSTM methods --------------------- #
    def lstm_tag(self, sentence: torch.Tensor) -> torch.Tensor:
        if self.mode!= "lstm":
            raise RuntimeError("lstm_tag is only available in lstm mode")
        return self.lstm_tagger(sentence)

    # --------------------- Classifier methods --------------------- #
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode!= "classifier":
            raise RuntimeError("classify is only available in classifier mode")
        return self.classifier(x)

    # --------------------- Convenience --------------------- #
    def __repr__(self) -> str:
        return f"<GraphQNNGen mode={self.mode}>"


__all__ = [
    "GraphQNNGen",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "build_classifier_circuit",
    "LSTMTagger",
    "ClassicalQLSTM",
]
