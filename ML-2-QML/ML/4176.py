"""Graph-based hybrid neural network that unifies classical GNNs, fraud‑detection style layers,
and quantum‑inspired LSTM cells.

The public API mirrors the original GraphQNN module but extends it with:
* A `GraphQNNHybrid` class that can operate purely classically or with a quantum‑inspired LSTM.
* Fraud‑layer construction (`FraudLayerParameters` and `_layer_from_params`) adapted from the fraud‑detection seed.
* A lightweight LSTM implementation (`QLSTM` and `LSTMTagger`) that can be swapped for a quantum version.
* Utility functions for random network generation, training data, state fidelity, and adjacency construction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1. Fraud‑Detection style layer utilities
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


# --------------------------------------------------------------------------- #
# 2. Classical LSTM / Quantum‑inspired LSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
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
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None,
    ) -> Tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between :class:`QLSTM` or ``nn.LSTM``."""

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
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: Tensor) -> Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
# 3. GraphQNNHybrid – core class
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """
    Unified graph neural network that optionally augments node embeddings with
    a quantum‑inspired LSTM.  It exposes the same public API as the original
    GraphQNN module but also accepts fraud‑layer parameters and can switch
    between classical and quantum LSTM back‑ends.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_lstm: bool = False,
        n_qubits: int = 0,
        lstm_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.weights = self._init_weights()
        self.use_lstm = use_lstm
        if use_lstm:
            if n_qubits <= 0:
                raise ValueError("n_qubits must be positive when use_lstm=True.")
            self.lstm = QLSTM(
                input_dim=qnn_arch[-1],
                hidden_dim=lstm_hidden_dim or qnn_arch[-1],
                n_qubits=n_qubits,
            )
        else:
            self.lstm = None

    # --------------------------------------------------------------------- #
    # 3.1 Weight helpers
    # --------------------------------------------------------------------- #
    def _init_weights(self) -> List[Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        return weights

    # --------------------------------------------------------------------- #
    # 3.2 Random network & training data
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = GraphQNNHybrid.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    # --------------------------------------------------------------------- #
    # 3.3 Forward propagation
    # --------------------------------------------------------------------- #
    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            if self.use_lstm:
                # Run the quantum‑inspired LSTM on the final embedding
                seq = current.unsqueeze(0).unsqueeze(0)  # (seq_len=1, batch=1, dim)
                lstm_out, _ = self.lstm(seq)
                activations.append(lstm_out.squeeze(0).squeeze(0))
            stored.append(activations)
        return stored

    # --------------------------------------------------------------------- #
    # 3.4 Graph utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor], threshold: float,
        *, secondary: float | None = None, secondary_weight: float = 0.5
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
    # 3.5 Fraud‑layer construction (public API)
    # --------------------------------------------------------------------- #
    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> nn.Sequential:
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)


__all__ = [
    "FraudLayerParameters",
    "QLSTM",
    "LSTMTagger",
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
