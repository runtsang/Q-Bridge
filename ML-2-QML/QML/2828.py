"""Hybrid LSTM implementation that uses quantum circuits for the gates.

This module mirrors the classical version but replaces each gate with a
small variational quantum circuit.  The design is inspired by the
quantum seed (QLSTM.py) and the quantum regression example to provide a
high‑level interface that can be used interchangeably with the classical
implementation.

Key features:
* Each gate is a `tq.QuantumModule` that encodes the classical input
  into a quantum state, applies a trainable parameterised circuit,
  and measures all qubits.
* A small `RandomLayer` seeds the circuit with entanglement, followed
  by trainable RX/RY rotations, providing expressive variational
  capacity with a modest number of parameters.
* The output of the quantum circuit is a real‑valued feature vector
  that is passed through a linear layer to obtain the gate activation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Quantum LSTM cell where each gate is realised by a variational circuit."""
    class QGate(tq.QuantumModule):
        """Encodes a classical vector into a quantum state and applies a trainable circuit."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode each input dimension as an Ry rotation on a separate wire.
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            # Random entangling layer
            self.random = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Gate circuits
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate  = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Linear maps from classical concatenated state to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Post‑measurement linear head to produce gate activations
        self.forget_head = nn.Linear(n_qubits, hidden_dim)
        self.input_head  = nn.Linear(n_qubits, hidden_dim)
        self.update_head = nn.Linear(n_qubits, hidden_dim)
        self.output_head = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical to qubit encoding
            f_q = self.forget_lin(combined)
            i_q = self.input_lin(combined)
            g_q = self.update_lin(combined)
            o_q = self.output_lin(combined)

            # Quantum gates
            bsz = f_q.shape[0]
            qdev_f = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=f_q.device)
            qdev_i = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=i_q.device)
            qdev_g = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=g_q.device)
            qdev_o = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=o_q.device)

            f_enc = self.forget_gate(qdev_f)
            i_enc = self.input_gate(qdev_i)
            g_enc = self.update_gate(qdev_g)
            o_enc = self.output_gate(qdev_o)

            f = torch.sigmoid(self.forget_head(f_enc))
            i = torch.sigmoid(self.input_head(i_enc))
            g = torch.tanh(self.update_head(g_enc))
            o = torch.sigmoid(self.output_head(o_enc))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can swap between the classical and quantum LSTM."""
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
