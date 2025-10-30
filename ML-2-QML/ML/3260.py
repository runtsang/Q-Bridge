"""Hybrid LSTM implementation with optional quantum‑inspired gates and a configurable classifier head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier with ``depth`` hidden layers.
    Returns the network, indices of the inputs, weight sizes and output
    observables placeholders.  The API mirrors the quantum helper but stays
    purely classical.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
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
    observables: List[int] = list(range(2))
    return network, encoding, weight_sizes, observables


class ClassicalQLSTMCell(nn.Module):
    """
    Classical LSTM cell that mimics the quantum gate structure.
    Each gate is a linear projection followed by the standard activation.
    """
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

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_linear(combined))
        i = torch.sigmoid(self.input_linear(combined))
        g = torch.tanh(self.update_linear(combined))
        o = torch.sigmoid(self.output_linear(combined))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


class HybridQLSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical LSTM cell
    and a quantum‑inspired variant.  It also exposes a configurable
    classifier head that can be used for downstream tasks.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        classifier_depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm_cell = ClassicalQLSTMCell(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        self.classifier, _, _, _ = build_classifier_circuit(hidden_dim, classifier_depth)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        ``sentence`` is a 1‑D tensor of token indices.
        Returns log‑softmaxed tag logits for each token.
        """
        embeds = self.word_embeddings(sentence)
        hx = torch.zeros(sentence.size(0), self.hidden_dim, device=embeds.device)
        cx = torch.zeros(sentence.size(0), self.hidden_dim, device=embeds.device)

        outputs: List[torch.Tensor] = []
        for x in embeds:
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))

        lstm_out = torch.cat(outputs, dim=0)
        logits = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
        logits = logits.view(len(sentence), -1, self.hidden2tag.out_features)
        return F.log_softmax(logits, dim=2)

__all__ = ["HybridQLSTMTagger", "build_classifier_circuit", "ClassicalQLSTMCell"]
