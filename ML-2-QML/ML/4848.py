"""Hybrid classical model combining classifier, LSTM tagger, and regression."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# ----------------------------------------------------------------------
# 1. Classical classifier
# ----------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int, use_batchnorm: bool = False, dropout: float = 0.0) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier with optional batch‑norm and dropout.
    Returns the network, a list of feature indices (encoding), a list of
    parameter counts per layer, and observable indices for compatibility with
    the quantum counterpart.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(num_features))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# ----------------------------------------------------------------------
# 2. Classical / placeholder quantum LSTM
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Hybrid LSTM cell that can operate in a pure classical mode or a
    quantum‑enhanced mode via the QML side.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical linear projections for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gates are placeholders when use_quantum is False
        if use_quantum and n_qubits > 0:
            self.forget_gate = nn.Identity()
            self.input_gate = nn.Identity()
            self.update_gate = nn.Identity()
            self.output_gate = nn.Identity()
        else:
            self.forget_gate = nn.Identity()
            self.input_gate = nn.Identity()
            self.update_gate = nn.Identity()
            self.output_gate = nn.Identity()

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum and n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, use_quantum=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# ----------------------------------------------------------------------
# 3. Classical regression dataset and model
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

# ----------------------------------------------------------------------
# 4. Unified wrapper
# ----------------------------------------------------------------------
class HybridModel(nn.Module):
    """
    Wrapper that selects the appropriate sub‑module based on ``mode``.
    Modes: 'classifier','regression', 'tagger'.
    """
    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == "classifier":
            num_features = kwargs["num_features"]
            depth = kwargs["depth"]
            self.circuit, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)
        elif mode == "regression":
            num_features = kwargs["num_features"]
            self.model = QModel(num_features)
        elif mode == "tagger":
            embedding_dim = kwargs["embedding_dim"]
            hidden_dim = kwargs["hidden_dim"]
            vocab_size = kwargs["vocab_size"]
            tagset_size = kwargs["tagset_size"]
            n_qubits = kwargs.get("n_qubits", 0)
            self.model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=n_qubits)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward(self, *args, **kwargs):
        if self.mode == "classifier":
            raise NotImplementedError("Classifier mode forward is not implemented in the high‑level wrapper.")
        else:
            return self.model(*args, **kwargs)

__all__ = [
    "build_classifier_circuit",
    "QLSTM",
    "LSTMTagger",
    "RegressionDataset",
    "QModel",
    "HybridModel",
    "generate_superposition_data",
]
