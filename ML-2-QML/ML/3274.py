"""Hybrid classical LSTM with a feed‑forward classifier head.

This module mirrors the original QLSTM.py but adds a simple
classifier built by :func:`build_classifier_circuit`.  The
:class:`QLSTM` is a drop‑in classical replacement for the quantum
version, and :class:`LSTMTagger` can switch between the classical LSTM
and the quantum‑inspired variant.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a classical feed‑forward classifier mirroring the quantum interface.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector (typically the hidden state dimension).
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Sequential
        The classifier network.
    encoding : list[int]
        Indices of the input features used for encoding (placeholder for API compatibility).
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    observables : list[int]
        Dummy list of observables (present for API compatibility with the quantum variant).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for API compatibility
    return network, encoding, weight_sizes, observables


class QLSTM(nn.Module):
    """Classical LSTM cell that mimics the quantum interface."""

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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the classical LSTM
    and the quantum‑inspired LSTM, and uses a classical feed‑forward
    classifier head.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Classical classifier head
        self.classifier, _, _, _ = build_classifier_circuit(
            num_features=hidden_dim,
            depth=classifier_depth,
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        hidden = lstm_out.view(len(sentence), -1)

        # Map hidden state to tag logits via the classifier
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "build_classifier_circuit"]
