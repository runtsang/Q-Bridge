from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Iterable

# ------------------------------------------------------------------
# Classical LSTM implementation with optional quantum flag
# ------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Classical LSTM that can be extended to use quantum gates.
    When ``use_quantum`` is True a quantum implementation must
    replace the linear gates externally; the class raises a
    RuntimeError to remind the user.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, use_quantum: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum and n_qubits > 0

        if self.use_quantum:
            # Placeholder – the quantum module must be swapped in later.
            self.forget = None
            self.input_gate = None
            self.update = None
            self.output = None
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the classical QLSTM by default.
    The constructor accepts ``use_quantum=True`` to signal the caller
    that a quantum version should be swapped in later.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim,
                          n_qubits=n_qubits, use_quantum=use_quantum)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# ------------------------------------------------------------------
# Classical regression utilities
# ------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a
    non‑linear function of the sum of input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}


class QModel(nn.Module):
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
        return self.net(state_batch).squeeze(-1)


# ------------------------------------------------------------------
# Fully‑connected layer emulation
# ------------------------------------------------------------------
def FCL():
    """
    Return a classical substitute for the quantum fully‑connected layer
    that exposes a ``run`` method compatible with the quantum version.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()
    return FullyConnectedLayer()


# ------------------------------------------------------------------
# Classical classifier factory
# ------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int):
    """
    Construct a purely classical feed‑forward classifier and return
    the network together with metadata that mirrors the quantum helper.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

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


__all__ = ["QLSTM", "LSTMTagger", "RegressionDataset",
           "generate_superposition_data", "QModel", "FCL",
           "build_classifier_circuit"]
