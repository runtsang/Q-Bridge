"""HybridClassifier and hybrid LSTM tagger for classical networks.

This module defines:
  * :class:`HybridClassifier` – a deep residual feed‑forward classifier
    that mirrors the quantum circuit builder interface.
  * :class:`ClassicalQLSTM` – a standard LSTM cell with linear gates.
  * :class:`LSTMTagger` – a sequence tagging model that can switch
    between a classical LSTM and the quantum placeholder.

The build_classifier_circuit static method returns a tuple
(network, encoding, weight_sizes, observables) exactly as the
original anchor file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class HybridClassifier(nn.Module):
    """Deep residual feed‑forward classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of input features.
    depth : int
        Number of residual blocks.
    hidden_dim : int | None
        Width of hidden layers; defaults to ``num_features``.
    """

    def __init__(self, num_features: int, depth: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features

        layers: List[nn.Module] = []
        in_dim = num_features
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, self.hidden_dim)
            bn = nn.BatchNorm1d(self.hidden_dim)
            relu = nn.ReLU()
            layers.append(nn.Sequential(linear, bn, relu))
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.hidden_dim

        # Final head
        head = nn.Linear(self.hidden_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.observables = list(range(2))

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return a fully‑connected residual network and metadata."""
        instance = HybridClassifier(num_features, depth)
        return instance.network, instance.encoding, instance.weight_sizes, instance.observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual network."""
        out = x
        for block in self.network:
            out = block(out) + out  # residual skip
        return out

class ClassicalQLSTM(nn.Module):
    """Standard LSTM cell with optional pre‑processing linear transforms."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either a classical or a quantum LSTM."""

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # For ML side, we use a placeholder classical LSTM
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridClassifier", "ClassicalQLSTM", "LSTMTagger"]
