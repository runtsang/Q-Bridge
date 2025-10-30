"""Unified kernel, classifier, and optional LSTM tagger for classical ML pipelines."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ClassicalRBFKernel(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel:
    """Wrapper for classical or quantum kernel."""
    def __init__(self, use_quantum: bool = False, gamma: float = 1.0):
        self.use_quantum = use_quantum
        self.kernel = ClassicalRBFKernel(gamma) if not use_quantum else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            raise NotImplementedError("Quantum kernel is not available in the classical module.")
        return self.kernel.forward(x, y)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

def build_classifier(num_features: int, depth: int, *, use_quantum: bool = False) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[Any]]:
    if not use_quantum:
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables
    else:
        raise NotImplementedError("Quantum classifier is not available in the classical module.")

class QLSTM(nn.Module):
    """Classical LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model with classical LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim) if n_qubits == 0 else QLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridKernelClassifier:
    """Unified hybrid kernel, classifier, and optional LSTM tagger."""
    def __init__(self, num_features: int, depth: int, gamma: float = 1.0, n_qubits: int = 0):
        self.kernel = Kernel(use_quantum=False, gamma=gamma)
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier(num_features, depth, use_quantum=False)
        self.lstm = QLSTM(num_features, depth, n_qubits=n_qubits)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.gram_matrix(a, b)

    def tag(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.lstm(sentence)

__all__ = ["HybridKernelClassifier", "Kernel", "build_classifier", "QLSTM", "LSTMTagger"]
