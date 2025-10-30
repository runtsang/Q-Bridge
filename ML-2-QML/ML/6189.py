"""Hybrid classical LSTM with optional quantum‑inspired gate layers.

This module provides a QLSTM class that can be configured to use
purely linear gates or to replace each gate with a lightweight
fully‑connected quantum‑inspired layer.  The implementation is
fully classical and relies only on PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable

class FullyConnectedLayer(nn.Module):
    """Lightweight quantum‑inspired fully connected layer.

    The layer maps a 1‑D tensor of parameters to a single scalar
    expectation value using a tanh activation.  It mimics the
    behaviour of the quantum FCL example while remaining purely
    classical.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.tensor(list(thetas), dtype=torch.float32,
                              device=self.linear.weight.device).view(-1, 1)
        return torch.tanh(self.linear(values)).mean()

class QLSTM(nn.Module):
    """Classical LSTM with optional quantum‑inspired gate layers."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 use_qclayer: bool = False,
                 n_features: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_qclayer = use_qclayer
        self.n_features = n_features

        gate_dim = hidden_dim
        if use_qclayer:
            # Each gate is a FullyConnectedLayer that outputs a scalar
            # which is broadcast to the hidden dimension.
            self.forget = FullyConnectedLayer(n_features)
            self.input_g = FullyConnectedLayer(n_features)
            self.update = FullyConnectedLayer(n_features)
            self.output = FullyConnectedLayer(n_features)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_features)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_features)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_features)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_features)
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.use_qclayer:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input_g(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid QLSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_qclayer: bool = False,
                 n_features: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim,
                          use_qclayer=use_qclayer,
                          n_features=n_features)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger", "FullyConnectedLayer"]
