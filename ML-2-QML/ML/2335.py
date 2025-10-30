import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Tuple, Iterable, Sequence, List

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell with optional quantum‑inspired gate activations."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

class QLSTMGraphQNN(nn.Module):
    """Hybrid LSTM‑Graph module that can operate in classical or quantum mode."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # In the classical module we simulate quantum gates with linear layers
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.use_graph = use_graph

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def compute_fidelity_graph(
        self,
        hidden_states: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise cosine similarities of hidden states."""
        if not self.use_graph:
            raise RuntimeError("Graph construction disabled for this instance.")
        graph = nx.Graph()
        graph.add_nodes_from(range(hidden_states.size(0)))
        norms = hidden_states.norm(p=2, dim=1, keepdim=True) + 1e-12
        normalized = hidden_states / norms
        sims = normalized @ normalized.t()
        for i in range(hidden_states.size(0)):
            for j in range(i + 1, hidden_states.size(0)):
                fid = sims[i, j].item()
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

# ---- Utility functions ----

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random linear network and corresponding training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f))
    target_weight = weights[-1]
    training_data = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1))
        target = target_weight @ features
        training_data.append((features, target))
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Forward pass through a purely linear network, recording activations."""
    activations = []
    for features, _ in samples:
        layer_outputs = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_outputs.append(current)
        activations.append(layer_outputs)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine‑based fidelity between two vectors."""
    a_norm = a / (a.norm() + 1e-12)
    b_norm = b / (b.norm() + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build an adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "QLSTMGraphQNN",
    "ClassicalQLSTM",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
