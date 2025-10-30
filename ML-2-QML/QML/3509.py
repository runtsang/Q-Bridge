"""Quantum LSTM with variational gates and a fully‑connected quantum layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QFLayer(tq.QuantumModule):
    """Quantum‑parameterized fully‑connected layer producing expectation values."""
    def __init__(self, input_dim: int, output_dim: int, n_wires: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_wires = n_wires or output_dim
        self.param_linear = nn.Linear(input_dim, self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.param_linear(x)  # (batch, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        for i in range(self.n_wires):
            tqf.rx(qdev, params[:, i], wires=i)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        return self.measure(qdev)

class QLSTMGen130(nn.Module):
    """Quantum LSTM where each gate is a variational quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits or hidden_dim
        self.forget_gate = QFLayer(input_dim + hidden_dim, hidden_dim, n_wires=self.n_qubits)
        self.input_gate  = QFLayer(input_dim + hidden_dim, hidden_dim, n_wires=self.n_qubits)
        self.update_gate = QFLayer(input_dim + hidden_dim, hidden_dim, n_wires=self.n_qubits)
        self.output_gate = QFLayer(input_dim + hidden_dim, hidden_dim, n_wires=self.n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTaggerGen130(nn.Module):
    """Sequence tagging model using the quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int | None = None,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen130(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen130", "LSTMTaggerGen130"]
