"""
Unified hybrid model combining classical CNN, fully connected head and a classical LSTM cell.
The module is intentionally free of any quantum dependencies so it can be used on any
PyTorch‑enabled device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNN(nn.Module):
    """
    Classical CNN backbone followed by a linear head.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.head(features)
        return self.norm(out)


class ClassicalLSTMCell(nn.Module):
    """
    Standard LSTM cell implemented with linear gates.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.linear_forget(combined))
            i = torch.sigmoid(self.linear_input(combined))
            g = torch.tanh(self.linear_update(combined))
            o = torch.sigmoid(self.linear_output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the classical LSTM cell.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = ClassicalLSTMCell(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridCNN", "ClassicalLSTMCell", "LSTMTagger"]
