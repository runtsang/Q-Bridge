"""Quantum-enhanced LSTM with trainable gates.

This module implements a hybrid LSTM cell where the gate activations are
produced by a small variational quantum circuit.  The circuit consists
of:

* a GeneralEncoder that applies input‑dependent RX rotations,
* a trainable RX gate on each wire,
* a linear‑circuit of CNOTs that entangles the wires,
* measurement of all wires in the Pauli‑Z basis.

The cell supports:

* `batch_first` – input tensors of shape (batch, seq_len, feat) are
  accepted.
* `dropout` – optional dropout on the hidden state.
* `n_qubits` – number of qubits per gate.  The same number is used
  for all four gates.

The implementation relies on torchquantum and is fully differentiable
with respect to the circuit parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Parameterised quantum layer that maps a real vector to a real
    vector of the same dimension via a small variational circuit."""

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

        # Encode each input dimension into a separate RX rotation.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )

        # Trainable RX gates on each wire.
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        # Measure all wires in the Pauli‑Z basis.
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a quantum device with batch size equal to the batch
        # dimension of x.
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=x.shape[0],
            device=x.device,
        )

        # Encode the classical input.
        self.encoder(qdev, x)

        # Apply the trainable RX gates.
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)

        # Entangle the wires with a linear chain of CNOTs.
        for wire in range(self.n_wires):
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tqf.cnot(qdev, wires=[wire, tgt])

        # Return the expectation values of Pauli‑Z.
        return self.measure(qdev)


class QLSTM(nn.Module):
    """LSTM cell where each gate is implemented by a QLayer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        # Quantum layers for the four gates.
        self.forget_q = QLayer(n_qubits)
        self.input_q = QLayer(n_qubits)
        self.update_q = QLayer(n_qubits)
        self.output_q = QLayer(n_qubits)

        # Linear maps from (input + hidden) to the number of qubits.
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1 if self.batch_first else 0)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_q(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_q(self.input_linear(combined)))
            g = torch.tanh(self.update_q(self.update_linear(combined)))
            o = torch.sigmoid(self.output_q(self.output_linear(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
                batch_first=batch_first,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=batch_first,
                dropout=dropout,
            )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QLSTM", "LSTMTagger"]
