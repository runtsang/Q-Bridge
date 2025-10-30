"""Hybrid quantum LSTM that uses a trainable quantum sampler to generate gate probabilities."""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSampler(tq.QuantumModule):
    """Parameterized circuit that outputs probabilities for forget and input gates."""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = hidden_dim * 2  # first half for forget, second half for input
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, f_angles: torch.Tensor, i_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f_angles, i_angles: shape (batch, hidden_dim)
        Returns: probabilities for each gate, shape (batch, hidden_dim)
        """
        batch_size = f_angles.shape[0]
        device = f_angles.device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=device)
        # Apply rotations for forget gates
        for idx in range(self.hidden_dim):
            tq.RX(f_angles[:, idx], wires=idx)(qdev)
        # Apply rotations for input gates
        for idx in range(self.hidden_dim):
            tq.RX(i_angles[:, idx], wires=self.hidden_dim + idx)(qdev)
        # Optionally, add entangling gates to enrich correlations
        for idx in range(self.hidden_dim - 1):
            tqf.cnot(qdev, wires=[idx, idx + 1])
        probs = self.measure(qdev)  # expectation values in [-1,1]
        probs = (probs + 1) / 2  # convert to [0,1]
        f = probs[:, :self.hidden_dim]
        i = probs[:, self.hidden_dim:]
        return f, i

class QLSTM(nn.Module):
    """Quantum-enhanced LSTM where forget and input gates are sampled from a quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sampler = QuantumSampler(hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_lin = self.forget_linear(combined)
            i_lin = self.input_linear(combined)
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            f, i = self.sampler(f_lin, i_lin)
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
    """Tagger using the hybrid quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
