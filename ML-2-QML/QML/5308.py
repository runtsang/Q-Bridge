"""Hybrid quantum Graph Neural Network combining quanvolution, quantum LSTM, and photonic fraud detection.

This module mirrors the classical implementation but replaces key components with quantum circuits.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import qiskit
from qiskit.circuit.random import random_circuit
import qutip as qt
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf

# Import utilities from the original quantum seed modules
from GraphQNN import feedforward, fidelity_adjacency, random_network, state_fidelity
from Conv import Conv
from QLSTM import QLSTM, LSTMTagger
from FraudDetection import FraudLayerParameters, build_fraud_detection_program

Tensor = torch.Tensor

class PhotonicFraudHead(nn.Module):
    """Wrapper that runs a Strawberry Fields program and returns a scalar tensor."""

    def __init__(self, program: sf.Program):
        super().__init__()
        self.program = program
        self.engine = sf.Engine("fock", backend_options={"cutoff_dim": 10})

    def forward(self, x: Tensor) -> Tensor:
        # Run the program; ignore input x
        result = self.engine.run(self.program)
        # For demonstration, return the mean photon number of the first mode
        mean_photon = result.state.full()[0, 0].real
        return torch.tensor(mean_photon, dtype=torch.float32, device=x.device)

class GraphQNNHybrid(nn.Module):
    """Hybrid quantum graph neural network with quanvolution, quantum LSTM, and photonic fraud detection."""

    def __init__(
        self,
        arch: Sequence[int],
        fraud_params: List[FraudLayerParameters],
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold

        # Build quantum network layers
        _, self.unitaries, self.training_data, self.target_unitary = random_network(arch, samples=100)

        # Quantum convolution filter
        self.conv = Conv()

        # Quantum LSTM tagger
        embedding_dim = arch[0]
        hidden_dim = arch[1] if len(arch) > 1 else 64
        vocab_size = 128  # placeholder
        tagset_size = 10  # placeholder
        self.lstm_tagger = LSTMTagger(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            n_qubits=4,
        )

        # Photonic fraud detection head
        program = build_fraud_detection_program(fraud_params[0], fraud_params[1:])
        self.fraud_head = PhotonicFraudHead(program)

        self.fraud_params = fraud_params

    def forward(self, node_features: Tensor) -> Tensor:
        """
        Forward pass through the hybrid quantum network.

        Parameters
        ----------
        node_features : Tensor
            A tensor of shape (num_nodes, feature_dim) representing graph node features.
        """
        # 1. Quantum feedforward across layers
        activations = feedforward(self.arch, self.unitaries, [(node_features, None)])

        # 2. Apply quantum convolution filter to each node (using first node as demo)
        conv_out = torch.tensor(self.conv.run(node_features[0].cpu().numpy()), dtype=torch.float32)

        # 3. Quantum LSTM tagging on flattened sequence of node embeddings
        seq = activations[-1].flatten().unsqueeze(0)  # dummy sequence
        lstm_out, _ = self.lstm_tagger(seq)

        # 4. Photonic fraud detection head on final embedding
        fraud_out = self.fraud_head(lstm_out.squeeze())

        return conv_out + fraud_out

    def fidelity_graph(self, states: Sequence[qt.Qobj]) -> nx.Graph:
        """Return a graph of node states based on quantum fidelity."""
        return fidelity_adjacency(states, self.threshold)

__all__ = ["GraphQNNHybrid"]
