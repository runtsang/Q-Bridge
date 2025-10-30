"""Hybrid LSTM with optional quantum-like fully connected layer.

This module provides a drop-in replacement for the original
QLSTM that can be used in environments without quantum backends.
The `QLSTMGen303` class contains a classical LSTM cell that
optionally uses a lightweight fully connected layer (`FCL`) to
simulate quantum behaviour.  The `LSTMTaggerGen303` class
wraps the LSTM and adds a final classification head that can
switch between a classical linear layer and a quantum-inspired
fully connected layer.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCL(nn.Module):
    """Classical stand‑in for a fully connected quantum layer.

    The interface mimics the quantum `FCL` from the QML seed:
    it exposes a `run(thetas)` method that accepts an iterable of
    parameters and returns a NumPy array of the expectation value.
    Internally it uses a single linear layer followed by a tanh and
    mean, exactly matching the behaviour of the quantum example
    but without any quantum execution.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()


class QLSTMGen303(nn.Module):
    """Classical LSTM cell that optionally uses a quantum‑like gate.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default 0
        When > 0 the cell will route its gates through a
        lightweight `FCL` to emulate quantum behaviour.
    """
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

        if n_qubits > 0:
            # Replace the sigmoid/tanh activations with a quantum‑like FCL
            self.quantum_gate = FCL(n_features=gate_dim)
        else:
            self.quantum_gate = None

    def _gate(self, linear: nn.Linear, combined: torch.Tensor, activation: str) -> torch.Tensor:
        """Apply a gate, optionally through the quantum‑like FCL."""
        raw = linear(combined)
        if self.quantum_gate is not None:
            # Use the FCL to compute a scalar expectation per batch element.
            # The FCL expects an iterable of parameters; we pass the raw tensor.
            # Reshape to 1D array per batch for simplicity.
            thetas = raw.detach().cpu().numpy().flatten()
            # The FCL returns a single expectation value; broadcast to batch.
            expectation = self.quantum_gate.run(thetas)
            return torch.full((raw.size(0), 1), expectation, device=raw.device)
        if activation == "sigmoid":
            return torch.sigmoid(raw)
        if activation == "tanh":
            return torch.tanh(raw)
        return raw

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self._gate(self.forget_linear, combined, "sigmoid")
            i = self._gate(self.input_linear, combined, "sigmoid")
            g = self._gate(self.update_linear, combined, "tanh")
            o = self._gate(self.output_linear, combined, "sigmoid")
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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


class LSTMTaggerGen303(nn.Module):
    """Sequence tagging model that can switch between a classical LSTM and a
    quantum‑like LSTM.  The final classification head can also be
    replaced by a `FCL` that emulates a quantum fully‑connected layer.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen303(embedding_dim, hidden_dim, n_qubits=n_qubits)
        # Final classification head
        if n_qubits > 0:
            self.hidden2tag = FCL(n_features=hidden_dim)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        if isinstance(self.hidden2tag, FCL):
            # FCL expects iterable of parameters; we convert each hidden state
            # to a scalar expectation via the same mechanism.
            # For simplicity, we apply the FCL to the mean of hidden states.
            thetas = lstm_out.mean(dim=0).detach().cpu().numpy().flatten()
            logits = self.hidden2tag.run(thetas)
            return torch.log_softmax(torch.tensor(logits), dim=1)
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen303", "LSTMTaggerGen303", "FCL"]
