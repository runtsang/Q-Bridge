"""Hybrid classical‑quantum LSTM with configurable quantum‑inspired classifier head.

This module defines a classical PyTorch implementation that mimics the
quantum LSTM interface while providing a classifier architecture
inspired by the quantum ansatz used in the QML reference.  The
classifier can be configured with an arbitrary depth, and the module
exposes metadata (parameter sizes, observable indices) that
facilitate cross‑validation with the quantum counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Build a multi‑layer feed‑forward classifier that mirrors the
    structure of the quantum circuit used in the QML reference.

    Returns
    -------
    nn.Module
        Sequential network.
    Iterable[int]
        Encoding indices (identical to the number of features).
    Iterable[int]
        Weight‑size metadata for each linear layer.
    list[int]
        Observable indices (used as dummy metadata for the quantum
        implementation).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
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
    observables = list(range(2))
    return network, list(range(num_features)), weight_sizes, observables


class QLSTM(nn.Module):
    """Classical LSTM cell with a quantum‑inspired classifier head.

    The gates are standard linear layers; the classifier is built
    via :func:`build_classifier_circuit` and can be tuned with the
    ``depth`` argument.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features=hidden_dim,
            depth=depth,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        hx, cx = self._init_states(inputs, states)
        outputs: list[torch.Tensor] = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        hidden_seq = torch.cat(outputs, dim=0)
        logits = self.classifier(hidden_seq)
        return hidden_seq, (hx, cx), logits

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the classical
    :class:`QLSTM` or a pure PyTorch :class:`nn.LSTM`.

    Parameters
    ----------
    embedding_dim : int
        Dimension of token embeddings.
    hidden_dim : int
        Hidden dimension of the LSTM.
    vocab_size : int
        Number of distinct tokens.
    tagset_size : int
        Number of tags in the output space.
    n_qubits : int, default 0
        If >0, a :class:`QLSTM` is instantiated; otherwise a
        standard :class:`nn.LSTM` is used.
    depth : int, default 2
        Depth of the classifier network inside :class:`QLSTM`.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, states, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
