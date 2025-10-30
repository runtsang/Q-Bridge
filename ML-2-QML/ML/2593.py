"""Unified Graph‑based LSTM model with classical graph processing.

This module implements a `UnifiedGraphQLSTM` class that
- builds a random graph from a classical random‑network generator.
- computes a fidelity‑based adjacency graph.
- runs a classical feed‑forward network on the graph nodes.
- sequences the node embeddings through a classical LSTMTagger.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# Import helpers from the original GraphQNN seed
from GraphQNN import feedforward, fidelity_adjacency, random_network

class SimpleLSTMTagger(nn.Module):
    """A minimal LSTM tagger that operates on feature tensors."""
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(x)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

class UnifiedGraphQLSTM(nn.Module):
    """Hybrid graph‑LSTM model that uses classical graph propagation
    followed by a classical sequence tagger.

    Parameters
    ----------
    graph_arch : Sequence[int]
        Architecture of the random graph neural network (layer sizes).
    lstm_hidden_dim : int
        Hidden dimension of the LSTM tagger.
    vocab_size : int
        Size of the vocabulary for embedding layer. (unused in the classical version)
    tagset_size : int
        Number of tags for classification.
    n_qubits : int
        Number of qubits used by the quantum LSTM in the QML version.
        Ignored in the classical implementation.
    fidelity_threshold : float
        Threshold for constructing the fidelity adjacency graph.
    """
    def __init__(
        self,
        graph_arch,
        lstm_hidden_dim,
        vocab_size,
        tagset_size,
        n_qubits=0,
        fidelity_threshold=0.9,
    ):
        super().__init__()
        # Build the random graph and training data
        self.random_arch, self.weights, self.training_data, self.target_weight = random_network(
            list(graph_arch), samples=100
        )
        # Compute the fidelity adjacency graph
        states = [self.target_weight]
        self.adjacency = fidelity_adjacency(states, fidelity_threshold)

        # Classical LSTM tagger
        self.lstm_tagger = SimpleLSTMTagger(
            input_dim=graph_arch[-1],
            hidden_dim=lstm_hidden_dim,
            tagset_size=tagset_size,
        )

    def forward(self, node_sequences: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_sequences : torch.Tensor
            Tensor of shape (seq_len, batch, feature_dim) representing
            a sequence of node embeddings for each batch element.

        Returns
        -------
        torch.Tensor
            Log‑softmax scores for each tag at each time step.
        """
        # Classical GNN feedforward
        seq_len, batch, feat_dim = node_sequences.shape
        flattened = node_sequences.reshape(seq_len * batch, feat_dim)
        activations = feedforward(
            self.random_arch,
            self.weights,
            [(flattened[i], None) for i in range(len(flattened))],
        )
        node_states = torch.stack([act[-1] for act in activations])
        node_states = node_states.reshape(seq_len, batch, -1)

        # Aggregate neighbor states using adjacency graph
        adj_mat = nx.to_numpy_array(self.adjacency)
        aggregated = torch.einsum(
            "ij, tjb -> tia",
            torch.tensor(adj_mat, dtype=node_states.dtype, device=node_states.device),
            node_states,
        )

        # Pass aggregated sequence through LSTM tagger
        logits = self.lstm_tagger(aggregated)
        return logits
