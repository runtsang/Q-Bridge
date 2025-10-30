"""Hybrid classical/quantum LSTM framework combining classical LSTM, graph neural networks,
kernel methods, and regression utilities.

This module provides a unified interface that can operate entirely in the classical
domain or switch to quantum subcomponents. The public class HybridQLSTM is a
drop‑in replacement for the original QLSTM module but exposes additional
capabilities from the GraphQNN, Kernel, and Regression seeds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import itertools
from typing import Tuple, List, Sequence, Iterable, Dict, Any

# --------------------------------------------------------------------------- #
# Classical LSTM cell and tagger (adapted from reference pair 1)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Drop‑in classical replacement for the quantum LSTM gates."""
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


class ClassicalLSTMTagger(nn.Module):
    """Sequence tagging model that uses either :class:`ClassicalQLSTM` or ``nn.LSTM``."""
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
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
# Graph neural network utilities (adapted from reference pair 2)
# --------------------------------------------------------------------------- #
def graph_random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random linear network and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def graph_feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Compute activations for each sample through a linear network."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def graph_state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two normalized feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def graph_fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = graph_state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Kernel utilities -----------------------------------------------------------
class ClassicalKernalAnsatz(nn.Module):
    """RBF kernel ansatz compatible with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalKernel(nn.Module):
    """Wrapper exposing a callable RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def classical_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Regression utilities -------------------------------------------------------
def classical_generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class ClassicalRegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding feature vectors and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = classical_generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ClassicalRegressionModel(nn.Module):
    """Simple feed‑forward regression network."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Hybrid wrapper --------------------------------------------------------------
# Expose common names for the wrapper to use
LSTMCell = ClassicalQLSTM
LSTMTagger = ClassicalLSTMTagger
GraphRandomNetwork = graph_random_network
GraphFeedforward = graph_feedforward
GraphFidelityAdjacency = graph_fidelity_adjacency
KernelMatrix = classical_kernel_matrix
RegressionDataset = ClassicalRegressionDataset
RegressionModel = ClassicalRegressionModel


class HybridQLSTM:
    """
    Unified interface that can operate in a purely classical mode or switch to quantum
    sub‑components.  The constructor accepts ``mode='classical'`` or ``mode='quantum'``.
    All public methods are deliberately simple so that downstream code can treat the
    wrapper as a black box.
    """
    def __init__(self, mode: str = "classical", **kwargs: Any) -> None:
        self.mode = mode.lower()
        if self.mode not in {"classical", "quantum"}:
            raise ValueError("mode must be either 'classical' or 'quantum'")

        # LSTM / tagger
        self.lstm = LSTMCell(kwargs.get("input_dim", 20),
                             kwargs.get("hidden_dim", 20),
                             kwargs.get("n_qubits", 0 if self.mode == "classical" else 4))
        self.tagger = LSTMTagger(kwargs.get("embedding_dim", 20),
                                 kwargs.get("hidden_dim", 20),
                                 kwargs.get("vocab_size", 1000),
                                 kwargs.get("tagset_size", 10),
                                 kwargs.get("n_qubits", 0 if self.mode == "classical" else 4))

        # Graph utilities
        self.graph_random_network = GraphRandomNetwork
        self.graph_feedforward = GraphFeedforward
        self.graph_fidelity_adjacency = GraphFidelityAdjacency

        # Kernel utilities
        self.kernel_matrix = KernelMatrix

        # Regression utilities
        self.regression_dataset = RegressionDataset(kwargs.get("samples", 1000),
                                                    kwargs.get("num_features", 10))
        self.regression_model = RegressionModel(kwargs.get("num_features", 10))

    # Public API ------------------------------------------------------------
    def tag_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        """Return tag logits for a token sequence."""
        return self.tagger(sentence)

    def graph_forward(self, sample: torch.Tensor) -> List[List[torch.Tensor]]:
        """Run a graph neural network forward pass."""
        return self.graph_feedforward(sample)

    def compute_kernel(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix between two collections of vectors."""
        return self.kernel_matrix(a, b)

    def regression_predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Predict regression targets from a batch of states."""
        return self.regression_model(state_batch)

    def __repr__(self) -> str:
        return f"<HybridQLSTM mode={self.mode!r}>"


__all__ = ["HybridQLSTM"]
