"""Hybrid quantum LSTMTagger that combines a quantum feature extractor
with a variational LSTM cell.  The implementation mirrors the classical
module but replaces all linear gate transformations with small variational
circuits that include a random layer and parametric rotations.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumRandomFeature(tq.QuantumModule):
    """Quantum feature map that encodes an input vector into a register of
    ``n_wires`` qubits.  A random layer followed by a few parametric
    rotations is used to enrich the representation.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        for wire, gate in enumerate(self.rx):
            gate(qdev, wires=wire)
        return self.measure(qdev)


class QuantumQLSTM(nn.Module):
    """Variational LSTM cell where each gate is realised by a small
    quantum circuit based on :class:`QuantumRandomFeature`.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QuantumRandomFeature(n_qubits)
        self.input = QuantumRandomFeature(n_qubits)
        self.update = QuantumRandomFeature(n_qubits)
        self.output = QuantumRandomFeature(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
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


class HybridQLSTMTagger(tq.QuantumModule):
    """Quantum sequence tagging model that augments embeddings with a
    quantum feature map before feeding them into a variational LSTM cell.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.quantum_embed = QuantumRandomFeature(n_qubits)
        self.linear_embed = nn.Linear(embedding_dim, n_qubits)

        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)

        seq_len, batch, _ = embeds.shape
        flat = embeds.reshape(-1, self.embedding_dim)
        proj = self.linear_embed(flat)  # (seq_len*batch, n_qubits)
        qfeat = self.quantum_embed(proj)  # (seq_len*batch, n_qubits)
        qfeat = qfeat.reshape(seq_len, batch, -1)

        lstm_out, _ = self.lstm(qfeat.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTMTagger"]
