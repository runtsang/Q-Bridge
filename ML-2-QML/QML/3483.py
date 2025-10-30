"""Quantum‑enhanced LSTM with variational circuits inspired by QLSTM and Quantum‑NAT.

The quantum module replaces each classical LSTM gate with a small variational quantum circuit.
It uses a RandomLayer followed by trainable RX, RY, RZ, and CRX gates, and measures
Pauli‑Z on all wires.  The resulting qubit‑state vector is linearly mapped back to the
hidden dimension.  The interface is identical to the classical ``QLSTMGen102`` so it can
be swapped in existing code.

Typical usage:

    model = QLSTMGen102(embedding_dim=100, hidden_dim=128, n_qubits=4)
    logits, _ = model(embeds, None)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional


class QLSTMGen102(tq.QuantumModule):
    """Quantum LSTM where each gate is a variational circuit.

    Parameters
    ----------
    input_dim : int
        Size of the input at each time step.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used to encode the gate activations.
    """

    class QGateLayer(tq.QuantumModule):
        """Variational circuit that turns classical activations into qubit states."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Random layer to entangle the wires
            self.random = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.ry = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.rz = nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)])
            # Two‑qubit entanglement
            self.crx = nn.ModuleList([tq.CRX(has_params=True, trainable=True) for _ in range(n_wires)])
            # Measurement
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            # Entangle wires
            self.random(qdev)
            for i in range(self.n_wires):
                self.rx[i](qdev, wires=i)
                self.ry[i](qdev, wires=i)
                self.rz[i](qdev, wires=i)
                # Entangle with next wire (circular)
                self.crx[i](qdev, wires=[i, (i + 1) % self.n_wires])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for LSTM components
        self.forget_gate = self.QGateLayer(n_qubits)
        self.input_gate = self.QGateLayer(n_qubits)
        self.update_gate = self.QGateLayer(n_qubits)
        self.output_gate = self.QGateLayer(n_qubits)

        # Classical linear projections to qubit space
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Linear mapping from qubit measurement back to hidden dimension
        self.measure_to_hidden = nn.Linear(n_qubits, hidden_dim)

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (seq_len, batch, input_dim).
        states : tuple of tensors or None
            Initial hidden and cell states.

        Returns
        -------
        outputs : torch.Tensor
            Sequence of hidden states (seq_len, batch, hidden_dim).
        (hx, cx) : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):  # x: (batch, input_dim)
            combined = torch.cat([x, hx], dim=1)  # (batch, input_dim + hidden_dim)

            # Quantum‑gate activations
            f_raw = self.forget_gate(self.forget_linear(combined))
            i_raw = self.input_gate(self.input_linear(combined))
            g_raw = self.update_gate(self.update_linear(combined))
            o_raw = self.output_gate(self.output_linear(combined))

            # Convert raw measurement to probabilities via sigmoid/tanh
            f = torch.sigmoid(f_raw)
            i = torch.sigmoid(i_raw)
            g = torch.tanh(g_raw)
            o = torch.sigmoid(o_raw)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


class LSTMTaggerGen102(nn.Module):
    """Drop‑in tagger that can use the quantum LSTM above."""
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
        self.lstm = QLSTMGen102(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), sentence.size(1), -1))
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QLSTMGen102", "LSTMTaggerGen102"]
