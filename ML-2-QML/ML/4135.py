"""Hybrid graph neural network combining a classical auto‑encoder and LSTM with fidelity‑based graph construction.

The module defines :class:`HybridGraphQLSTMNet`, a drop‑in replacement that
mirrors the public API of the original GraphQNN, QLSTM, and Autoencoder
modules while fusing their capabilities.  The design is intentionally
minimal – training loops, data loaders, and persistence are omitted
to keep the interface clean and reusable.

Key components
---------------
* :class:`AutoencoderNet` – fully‑connected auto‑encoder.
* :class:`ClassicalQLSTM` – classical LSTM cell that mimics the quantum interface.
* :func:`fidelity_adjacency` – constructs a weighted graph from latent
  state fidelities.
* :class:`HybridGraphQLSTMNet` – the hybrid architecture that
  encodes node features, aggregates neighbour information, propagates
  through an LSTM, and reconstructs the original features.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple, Sequence

import networkx as nx
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Auto‑encoder utilities
# --------------------------------------------------------------------------- #

class AutoencoderNet(nn.Module):
    """Simple fully‑connected auto‑encoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Classical LSTM cell (drop‑in replacement)
# --------------------------------------------------------------------------- #

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell that mimics the quantum interface."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
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


# --------------------------------------------------------------------------- #
# Fidelity utilities (copied from GraphQNN)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
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
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid Graph‑LSTM auto‑encoder
# --------------------------------------------------------------------------- #

class HybridGraphQLSTMNet(nn.Module):
    """Hybrid graph neural network that combines a classical auto‑encoder,
    a (classical or quantum) LSTM, and fidelity‑based adjacency graph
    construction.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw node features.
    hidden_dim : int
        Hidden size of the LSTM cell.
    latent_dim : int, default 32
        Bottleneck dimensionality of the auto‑encoder.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for the encoder/decoder.
    dropout : float, default 0.1
        Dropout probability inside the auto‑encoder.
    n_qubits : int, default 0
        If >0 a quantum LSTM is used; otherwise a classical LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.lstm = (
            ClassicalQLSTM(latent_dim, hidden_dim, n_qubits=n_qubits)
            if n_qubits > 0
            else nn.LSTM(latent_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid architecture.

        Parameters
        ----------
        node_features : torch.Tensor
            Tensor of shape (N, D) where N is the number of nodes.
        adjacency : torch.Tensor
            Adjacency matrix of shape (N, N).

        Returns
        -------
        torch.Tensor
            Reconstructed node features of shape (N, D).
        """
        # 1. Encode node features into a latent space
        latent = self.autoencoder.encode(node_features)  # (N, L)

        # 2. Aggregate neighbour latent states
        neighbour_latent = torch.matmul(adjacency, latent)  # (N, L)

        # 3. Concatenate encoded and neighbour states
        combined = torch.cat([latent, neighbour_latent], dim=-1)  # (N, 2L)

        # 4. Treat each node as a timestep and feed into the LSTM
        seq = combined.unsqueeze(1)  # (N, 1, 2L)
        lstm_out, _ = self.lstm(seq)  # (N, 1, H)
        lstm_out = lstm_out.squeeze(1)  # (N, H)

        # 5. Decode back to the original feature space
        recon = self.output_layer(lstm_out)  # (N, D)
        return recon

    def build_fidelity_graph(
        self,
        node_features: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph based on latent‑state fidelities."""
        latent = self.autoencoder.encode(node_features)
        states = [l.cpu() for l in latent]
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=secondary_weight
        )
