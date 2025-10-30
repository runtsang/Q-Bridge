"""Graph‑based hybrid quantum‑classical network for node‑level and sequence tasks.

The module exposes a single :class:`GraphQNN_LSTM` that merges
1. a classical graph neural network (GNN) backbone built from
   torch.nn.Linear layers (leveraging the original GraphQNN feedforward
   routine),
2. a quantum‑enhanced LSTM (QLSTM) that uses a small variational
   circuit for each gate, and
3. a fidelity‑based adjacency construction that can be used to
   initialise or regularise the graph.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Tuple, List, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Classical GNN utilities (inspired by GraphQNN.py)
# --------------------------------------------------------------------------- #

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate pairs (x, y) where y = W x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and training data for the last layer."""
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
    """Perform a forward pass of the linear GNN for each sample."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two classical vectors."""
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
    """Build a weighted graph from state fidelities."""
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
# Quantum LSTM (QLSTM) – the quantum core
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """Hybrid LSTM cell where each gate is a small variational quantum circuit.

    The cell can be dropped‑in‑for‑all‑classical training pipelines.
    """

    class _QLayer(tq.QuantumModule):
        """Gate‑specific quantum circuit used in the QLSTM."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encoder: linear rotation of each input qubit
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Parameterised gates: one RX per qubit
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: Tensor) -> Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=x.shape[0],
                device=x.device,
            )
            self.encoder(qdev, x)
            for idx, gate in enumerate(self.params):
                gate(qdev, wires=idx)
            # Entangle the qubits to mix information
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            # Final measurement
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear projections to match qubit dimension
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum layers for each gate
        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

    def forward(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[Tensor] = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None,
    ) -> Tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Hybrid Graph + LSTM model
# --------------------------------------------------------------------------- #

class GraphQNN_LSTM(nn.Module):
    """Hybrid model that processes graph nodes with a classical GNN
    and sequences with a quantum‑enhanced LSTM.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes for the GNN (must end with output dimension).
    lstm_hidden_dim : int
        Hidden size of the quantum LSTM.
    lstm_n_qubits : int
        Number of qubits used in each QLSTM gate.
    n_qubits_gnn : int, optional
        Number of qubits per node for the GNN embedding (default 1).
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        lstm_hidden_dim: int,
        lstm_n_qubits: int,
        n_qubits_gnn: int = 1,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_n_qubits = lstm_n_qubits

        # GNN weights
        self.gnn_weights: List[nn.Linear] = nn.ModuleList()
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            self.gnn_weights.append(nn.Linear(in_f, out_f))

        # Quantum LSTM
        self.lstm = QLSTM(
            input_dim=self.qnn_arch[-1],  # last GNN output feeds into LSTM
            hidden_dim=lstm_hidden_dim,
            n_qubits=lstm_n_qubits,
        )

        # Final classifier (example: node classification)
        self.classifier = nn.Linear(lstm_hidden_dim, 10)  # arbitrary output size

    # --------------------------------------------------------------------- #
    # GNN helpers
    # --------------------------------------------------------------------- #

    def gnn_forward(self, node_features: Tensor) -> List[Tensor]:
        """Forward pass through the classical GNN."""
        activations = [node_features]
        current = node_features
        for linear in self.gnn_weights:
            current = torch.tanh(linear(current))
            activations.append(current)
        return activations

    def build_fidelity_graph(
        self,
        node_states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Wraps :func:`fidelity_adjacency` for the GNN node states."""
        return fidelity_adjacency(
            node_states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )

    # --------------------------------------------------------------------- #
    # LSTM helpers
    # --------------------------------------------------------------------- #

    def lstm_forward(
        self,
        sequence: Tensor,
        states: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through the quantum LSTM."""
        return self.lstm(sequence, states)

    # --------------------------------------------------------------------- #
    # Full forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        node_features: Tensor,
        sequence: Tensor,
        states: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Process graph nodes and a sequence, returning node logits and
        sequence logits.

        Parameters
        ----------
        node_features : Tensor
            Shape ``(num_nodes, feature_dim)``.
        sequence : Tensor
            Shape ``(seq_len, batch, feature_dim)``.
        states : Tuple[Tensor, Tensor] | None
            Optional initial hidden/cell states for the LSTM.
        """
        # GNN forward
        gnn_states = self.gnn_forward(node_features)

        # Use the last GNN activation as input to the LSTM
        lstm_input = gnn_states[-1]

        # LSTM forward
        lstm_out, _ = self.lstm_forward(lstm_input, states)

        # Classification
        node_logits = self.classifier(gnn_states[-1])
        seq_logits = self.classifier(lstm_out)
        return node_logits, seq_logits

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNN_LSTM",
    "QLSTM",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
