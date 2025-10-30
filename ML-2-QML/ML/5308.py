"""Hybrid classical Graph Neural Network combining convolution, LSTM, and fraud detection layers.

This module builds upon the original GraphQNN utilities and extends them with
classical convolution filtering, LSTM tagging, and a fraud detection head.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import torch
from torch import nn

# Import utilities from the original seed modules
from GraphQNN import feedforward, fidelity_adjacency, random_network, state_fidelity
from Conv import Conv
from QLSTM import QLSTM, LSTMTagger
from FraudDetection import FraudLayerParameters, build_fraud_detection_program

Tensor = torch.Tensor

class GraphQNNHybrid(nn.Module):
    """Hybrid classical graph neural network with optional Conv, LSTM, and fraud detection layers."""

    def __init__(
        self,
        arch: Sequence[int],
        fraud_params: List[FraudLayerParameters],
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold

        # Build graph propagation layers
        _, self.weights, self.training_data, self.target_weight = random_network(arch, samples=100)

        # Convolution filter
        self.conv = Conv()

        # LSTM tagger
        embedding_dim = arch[0]
        hidden_dim = arch[1] if len(arch) > 1 else 64
        vocab_size = 128  # placeholder
        tagset_size = 10  # placeholder
        self.lstm_tagger = LSTMTagger(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            n_qubits=0,
        )

        # Fraud detection head
        self.fraud_head = build_fraud_detection_program(fraud_params[0], fraud_params[1:])

        self.fraud_params = fraud_params

    def forward(self, node_features: Tensor) -> Tensor:
        """
        Forward pass through the hybrid network.

        Parameters
        ----------
        node_features : Tensor
            A tensor of shape (num_nodes, feature_dim) representing graph node features.
        """
        # 1. Classical feedforward across layers
        activations = feedforward(self.arch, self.weights, [(node_features, None)])

        # 2. Apply convolution filter to each node (using first node as demo)
        conv_out = torch.tensor(self.conv.run(node_features[0].cpu().numpy()))

        # 3. LSTM tagging on flattened sequence of node embeddings
        seq = activations[-1].flatten().unsqueeze(0)  # dummy sequence
        lstm_out, _ = self.lstm_tagger(seq)

        # 4. Fraud detection head on final embedding
        fraud_out = self.fraud_head(lstm_out.squeeze())

        return conv_out + fraud_out

    def fidelity_graph(self, states: Sequence[Tensor]) -> nx.Graph:
        """Return a graph of node states based on classical fidelity."""
        return fidelity_adjacency(states, self.threshold)

__all__ = ["GraphQNNHybrid"]
