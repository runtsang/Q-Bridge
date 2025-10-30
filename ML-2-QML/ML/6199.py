"""Unified classical module that blends LSTM and graph‑based QNN utilities."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Dict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum library – falls back to None if unavailable
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:  # pragma: no cover
    tq = None
    tqf = None

# --------------------------------------------------------------------------- #
# 1. Random data generators (shared between classical and quantum)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: torch.Tensor, samples: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a dataset of (input, target) pairs for a linear target."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a random linear network and training data for its last weight."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# 2. LSTM / quantum‑enhanced LSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Hybrid LSTM cell that can use quantum variational gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum sub‑module (only built if n_qubits>0 and torchquantum is available)
        if n_qubits > 0 and tq is not None:
            self.forget_gate = self._make_gate(n_qubits)
            self.input_gate = self._make_gate(n_qubits)
            self.update_gate = self._make_gate(n_qubits)
            self.output_gate = self._make_gate(n_qubits)
        else:
            self.forget_gate = None
            self.input_gate = None
            self.update_gate = None
            self.output_gate = None

    # --------------------------------------------------------------------- #
    # Quantum sub‑module: small variational circuit for each gate
    # --------------------------------------------------------------------- #
    def _make_gate(self, n_qubits: int):
        """Return an nn.Module that runs a variational circuit on a vector."""
        class _Gate(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_qubits
                # Encode the input vector into rotation angles
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
                )
                # Trainable rotation gates
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                # simple entangling layer
                for i in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[i, i + 1])
                return self.measure(qdev)

        return _Gate()

    def _init_states(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch, self.hidden_dim, device=device)
        cx = torch.zeros(batch, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            if self.forget_gate is not None:
                f = torch.sigmoid(self.forget_gate(f))
                i = torch.sigmoid(self.input_gate(i))
                g = torch.tanh(self.update_gate(g))
                o = torch.sigmoid(self.output_gate(o))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # shape (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

# --------------------------------------------------------------------------- #
# 3. Graph‑based quantum neural network (classical simulation)
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Graph‑based QNN that propagates quantum‑like states through a unitary graph."""
    def __init__(self, qnn_arch: Sequence[int], n_qubits: int):
        self.arch = list(qnn_arch)
        self.n_qubits = n_qubits
        self.unitaries = self._init_unitaries()

    def _init_unitaries(self):
        """Create a list of random unitary layers, one per graph layer."""
        unitaries: List[List[torch.Tensor]] = [[]]
        for layer in range(1, len(self.arch)):
            ops: List[torch.Tensor] = []
            for _ in range(self.arch[layer]):
                if tq is not None:
                    ops.append(tq.RX(has_params=True, trainable=True))
                else:
                    ops.append(nn.Identity())
            unitaries.append(ops)
        return unitaries

    def feedforward(
        self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        stored_states = []
        for sample, _ in samples:
            current = sample
            layerwise = [current]
            for layer in range(1, len(self.arch)):
                for op in self.unitaries[layer]:
                    current = op(current)
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states

    def fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

# --------------------------------------------------------------------------- #
# 4. Unified wrapper
# --------------------------------------------------------------------------- #
class UnifiedQLSTMGraphQNN:
    """Composite model that couples a sequence tagger with a graph‑based QNN."""
    def __init__(self, lstm_config: Dict, graph_config: Dict):
        self.tagger = LSTMTagger(**lstm_config)
        self.graph = GraphQNN(**graph_config)

    def tag_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.tagger(sentence)

    def propagate_graph(
        self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    ):
        return self.graph.feedforward(samples)

    def graph_fidelity_adjacency(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        secondary: float | None = None,
    ) -> nx.Graph:
        return self.graph.fidelity_adjacency(states, threshold, secondary=secondary)

__all__ = ["QLSTM", "LSTMTagger", "GraphQNN", "UnifiedQLSTMGraphQNN"]
