"""Quantum‑enhanced LSTM where each gate is implemented by a variational circuit."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QGate(nn.Module):
    """A parameterised quantum circuit that returns a vector of qubit expectations.

    The circuit uses a single‑parameter per qubit rotation followed by a linear
    CNOT chain.  The number of qubits is fixed at construction time.
    """

    def __init__(self, n_qubits: int, feature_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.linear = nn.Linear(feature_dim, n_qubits, bias=False)

        dev = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            for i in range(n_qubits):
                qml.RX(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear mapping to qubit angles
        angles = self.linear(x)
        return self.circuit(angles)


class QLSTMPlus(nn.Module):
    """Quantum‑augmented LSTM where each gate is a :class:`QGate`."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        feature_dim = input_dim + hidden_dim

        # Quantum gates for each LSTM gate
        self.forget_gate = QGate(n_qubits, feature_dim)
        self.input_gate = QGate(n_qubits, feature_dim)
        self.update_gate = QGate(n_qubits, feature_dim)
        self.output_gate = QGate(n_qubits, feature_dim)

        # Optional linear post‑processing to match hidden_dim
        self.to_hidden = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.to_hidden(self.forget_gate(combined)))
            i = torch.sigmoid(self.to_hidden(self.input_gate(combined)))
            g = torch.tanh(self.to_hidden(self.update_gate(combined)))
            o = torch.sigmoid(self.to_hidden(self.output_gate(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

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
    """Sequence tagging model that uses the quantum :class:`QLSTMPlus`."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMPlus", "LSTMTagger"]
