"""Quantum‑enhanced LSTM with quantum kernel feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QKernelLayer(tq.QuantumModule):
    """Quantum circuit that maps a classical vector into a feature vector via expectation values."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.n_wires = input_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(input_dim)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(input_dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class QLSTM(nn.Module):
    """Quantum‑kernel driven LSTM where each gate is conditioned on quantum‑encoded features."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.kernel_layer = QKernelLayer(input_dim + hidden_dim)
        self.forget_linear = nn.Linear(self.kernel_layer.n_wires, hidden_dim)
        self.input_linear  = nn.Linear(self.kernel_layer.n_wires, hidden_dim)
        self.update_linear = nn.Linear(self.kernel_layer.n_wires, hidden_dim)
        self.output_linear = nn.Linear(self.kernel_layer.n_wires, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            k = self.kernel_layer(combined)  # quantum feature vector
            f = torch.sigmoid(self.forget_linear(k))
            i = torch.sigmoid(self.input_linear(k))
            g = torch.tanh(self.update_linear(k))
            o = torch.sigmoid(self.output_linear(k))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum‑kernel LSTM."""
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
