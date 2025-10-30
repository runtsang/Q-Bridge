"""
Hybrid LSTM module that unifies classical and quantum gate blocks.
"""

from __future__ import annotations

from typing import Tuple, Iterable, Sequence, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
# Classical gate layers
# --------------------------------------------------------------------------- #
class LinearGate(nn.Module):
    """Linear‑based gate that can be replaced by a quantum block."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# --------------------------------------------------------------------------- #
# Quantum gate placeholder
# --------------------------------------------------------------------------- #
class QuantumGateStub(nn.Module):
    """
    Placeholder for a quantum gate.  In the classical module we cannot
    instantiate real quantum circuits, so this stub simply applies a
    linear transformation and returns a vector of size `n_qubits`.
    """
    def __init__(self, in_dim: int, n_qubits: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# --------------------------------------------------------------------------- #
# Hybrid LSTM implementation
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    An LSTM cell that can optionally replace its four gates with quantum
    circuits.  When `n_qubits > 0` the cell uses a `QuantumGateStub`
    that pretends to be a quantum gate; the rest of the cell is identical
    to the classical LSTM implementation from the original seed.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_input = input_dim + hidden_dim
        self.forget_gate = QuantumGateStub(gate_input, n_qubits) if n_qubits > 0 else LinearGate(gate_input, hidden_dim)
        self.input_gate = QuantumGateStub(gate_input, n_qubits) if n_qubits > 0 else LinearGate(gate_input, hidden_dim)
        self.update_gate = QuantumGateStub(gate_input, n_qubits) if n_qubits > 0 else LinearGate(gate_input, hidden_dim)
        self.output_gate = QuantumGateStub(gate_input, n_qubits) if n_qubits > 0 else LinearGate(gate_input, hidden_dim)

        # Linear layers that feed the gates
        self.linear_forget = nn.Linear(gate_input, hidden_dim)
        self.linear_input = nn.Linear(gate_input, hidden_dim)
        self.linear_update = nn.Linear(gate_input, hidden_dim)
        self.linear_output = nn.Linear(gate_input, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
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
# Tagging model wrapper
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """
    Wrapper that can switch between a classical LSTM and the hybrid
    quantum‑enabled LSTM defined above.  The interface matches the
    original `QLSTM` seed.
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
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Classical classifier builder
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a simple feed‑forward classifier that mirrors the interface
    of the quantum `build_classifier_circuit`.  The returned tuple
    contains:
        * the network module,
        * a list of indices that would encode the input,
        * a list of indices that would hold trainable weights,
        * a list of output observables (here just dummy indices).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
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
# Simple regressor
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """
    Tiny fully‑connected regression network that mirrors the classical
    `EstimatorQNN` seed.  It is intentionally lightweight so that it can
    be used as a drop‑in replacement for the quantum version.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

# --------------------------------------------------------------------------- #
# Graph utilities (classical)
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

__all__ = [
    "HybridQLSTM",
    "LSTMTagger",
    "build_classifier_circuit",
    "EstimatorQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
