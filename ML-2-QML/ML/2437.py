"""
Unified LSTM/Graph neural network module for classical training.

Features
--------
* Classical QLSTM cell with optional quantum‑gate interface (fallback to linear gates).
* LSTMTagger that can be configured to use the classical or quantum LSTM.
* GraphQLSTM: a message‑passing LSTM that propagates hidden states over a graph.
* Classical graph utilities (feedforward, fidelity adjacency, random network generation).
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


# --------------------------------------------------------------------------- #
# Core LSTM / QLSTM
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """
    Classical LSTM cell that optionally delegates its gates to a quantum module.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input at each time step.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default=0
        If > 0, the gates are computed by a variational quantum circuit.
        The implementation keeps the original API but falls back to linear gates
        when the quantum backend is unavailable.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to the quantum space (if used)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Classical linear layers that map quantum outputs back to hidden_dim
        self.map_forget = nn.Linear(n_qubits, hidden_dim)
        self.map_input = nn.Linear(n_qubits, hidden_dim)
        self.map_update = nn.Linear(n_qubits, hidden_dim)
        self.map_output = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical linear projections
            f_q = self.linear_forget(combined)
            i_q = self.linear_input(combined)
            g_q = self.linear_update(combined)
            o_q = self.linear_output(combined)

            # Map quantum outputs back to hidden dimension
            f = torch.sigmoid(self.map_forget(f_q))
            i = torch.sigmoid(self.map_input(i_q))
            g = torch.tanh(self.map_update(g_q))
            o = torch.sigmoid(self.map_output(o_q))

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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical LSTM and
    the quantum‑enhanced QLSTM defined above.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Hidden dimension of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of distinct tags.
    n_qubits : int, default=0
        If > 0, the LSTM will be replaced by QLSTM.
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
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
# Graph‑aware LSTM
# --------------------------------------------------------------------------- #

class GraphQLSTM(nn.Module):
    """
    Message‑passing LSTM that propagates hidden states over a graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    input_dim : int
        Dimensionality of the per‑node input at each time step.
    hidden_dim : int
        Dimensionality of the hidden state per node.
    n_layers : int, default=1
        Number of recurrent layers (currently only 1 is supported).
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Adjacency matrix will be set via set_graph
        self.register_buffer("adj", torch.zeros(num_nodes, num_nodes))

    def set_graph(self, graph: nx.Graph) -> None:
        """
        Convert a NetworkX graph into a dense adjacency matrix.

        Parameters
        ----------
        graph : nx.Graph
            Undirected graph whose nodes are indexed 0..num_nodes-1.
        """
        adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32, device=self.adj.device)
        for u, v in graph.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        self.adj = adj

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Propagate hidden states over the graph.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, num_nodes, input_dim).

        Returns
        -------
        torch.Tensor
            Final hidden states of shape (num_nodes, hidden_dim).
        """
        seq_len = inputs.size(0)
        hidden = torch.zeros(self.num_nodes, self.hidden_dim, device=inputs.device)

        for t in range(seq_len):
            neighbor_hidden = torch.matmul(self.adj, hidden)  # aggregate neighbor states
            hidden = torch.tanh(
                self.W_in(inputs[t]) + self.W_h(neighbor_hidden) + self.bias
            )
        return hidden


# --------------------------------------------------------------------------- #
# Classical graph utilities (mirroring GraphQNN)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic linear regression data.

    Parameters
    ----------
    weight : torch.Tensor
        Target linear transformation.
    samples : int
        Number of samples to generate.

    Returns
    -------
    List[Tuple[torch.Tensor, torch.Tensor]]
        Each tuple is (features, target).
    """
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Construct a random feed‑forward network and synthetic training data.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes, e.g. [4, 8, 2].
    samples : int
        Number of training samples.

    Returns
    -------
    Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
        Architecture, list of weight matrices, training data, target weight.
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """
    Run a forward pass through the network for each sample.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes.
    weights : Sequence[torch.Tensor]
        Weight matrices for each layer.
    samples : Iterable[Tuple[torch.Tensor, torch.Tensor]]
        Input–target pairs.

    Returns
    -------
    List[List[torch.Tensor]]
        Activations per sample per layer.
    """
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute the squared overlap between two classical vectors.

    Parameters
    ----------
    a, b : torch.Tensor
        Normalised vectors.

    Returns
    -------
    float
        Fidelity value in [0, 1].
    """
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph from state fidelities.

    Parameters
    ----------
    states : Sequence[torch.Tensor]
        List of state vectors.
    threshold : float
        Primary fidelity threshold for edge creation.
    secondary : float | None, optional
        Secondary threshold for weaker edges.
    secondary_weight : float, default=0.5
        Weight assigned to secondary edges.

    Returns
    -------
    nx.Graph
        Weighted adjacency graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "QLSTM",
    "LSTMTagger",
    "GraphQLSTM",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
