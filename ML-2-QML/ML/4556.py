"""Hybrid classical layer combining fully connected, attention, LSTM‑style gating, and a classification head.

The design mirrors the quantum prototypes but remains purely classical, making it suitable for rapid prototyping and baseline comparisons.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLayer(nn.Module):
    """
    A hybrid classical layer that can act as a fully connected block,
    optionally augment with self‑attention, LSTM‑style gating, and a
    classification head.  The design mirrors the quantum prototypes
    while remaining purely classical.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_attention: bool = False,
        use_lstm_gate: bool = False,
        classifier_depth: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention
        self.use_lstm_gate = use_lstm_gate
        self.classifier_depth = classifier_depth

        # Fully connected core
        self.fc = nn.Linear(input_dim, hidden_dim)

        # Optional attention parameters
        if self.use_attention:
            self.attn_rot = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            self.attn_ent = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # Optional LSTM‑style gates
        if self.use_lstm_gate:
            gate_dim = hidden_dim
            self.forget = nn.Linear(hidden_dim + hidden_dim, gate_dim)
            self.input_gate = nn.Linear(hidden_dim + hidden_dim, gate_dim)
            self.update = nn.Linear(hidden_dim + hidden_dim, gate_dim)
            self.output_gate = nn.Linear(hidden_dim + hidden_dim, gate_dim)

        # Classification head
        if self.classifier_depth > 0:
            layers = []
            in_dim = hidden_dim
            for _ in range(self.classifier_depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 2))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional attention, LSTM gating, and classification.
        """
        # Base transformation
        h = self.fc(x)
        h = torch.tanh(h)

        # Optional attention
        if self.use_attention:
            query = h @ self.attn_rot
            key = h @ self.attn_ent
            scores = F.softmax(query @ key.t() / np.sqrt(self.hidden_dim), dim=-1)
            h = scores @ h

        # Optional LSTM gate
        if self.use_lstm_gate:
            hx = torch.zeros_like(h)
            cx = torch.zeros_like(h)
            outputs = []
            for step in range(h.size(0)):
                inp = h[step]
                combined = torch.cat([inp, hx], dim=-1)
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input_gate(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output_gate(combined))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            h = torch.cat(outputs, dim=0)

        # Classification
        if self.classifier is not None:
            h = self.classifier(h)

        return h


__all__ = ["HybridLayer"]
