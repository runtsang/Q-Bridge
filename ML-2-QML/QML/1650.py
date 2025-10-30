"""Quantum‑enhanced LSTM with variational gates, quantum‑dropout and layer‑norm.

The quantum implementation follows the classical API, enabling
drop‑in replacement while exposing genuinely quantum circuitry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """Quantum LSTM cell with optional quantum‑dropout and layer‑norm.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used for the variational gates.
    quantum_dropout : float, optional
        Dropout probability applied to the hidden state after each update.
    layer_norm : bool, optional
        If True, apply LayerNorm to the hidden state before computing gates.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        *,
        quantum_dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.quantum_dropout = quantum_dropout
        self.layer_norm = layer_norm

        # Quantum gates per LSTM component
        self.forget = self.QLayer(n_qubits, dropout=quantum_dropout)
        self.input = self.QLayer(n_qubits, dropout=quantum_dropout)
        self.update = self.QLayer(n_qubits, dropout=quantum_dropout)
        self.output = self.QLayer(n_qubits, dropout=quantum_dropout)

        # Classical linear projections into qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        if self.layer_norm:
            self.ln = nn.LayerNorm(hidden_dim)
        if self.quantum_dropout > 0.0:
            self.drop = nn.Dropout(self.quantum_dropout)

    class QLayer(tq.QuantumModule):
        """Variational circuit that maps a classical vector to a qubit measurement."""

        def __init__(self, n_wires: int, dropout: float = 0.0):
            super().__init__()
            self.n_wires = n_wires
            self.dropout = dropout
            # Simple encoder that applies RX gates conditioned on input features
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle all wires in a ring
            for wire in range(self.n_wires):
                tqf.cnot(qdev, wires=[wire, (wire + 1) % self.n_wires])
            out = self.measure(qdev)
            if self.dropout > 0.0:
                mask = torch.bernoulli((1 - self.dropout) * torch.ones_like(out)).to(out.device)
                out = out * mask
            return out

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
            if self.layer_norm:
                hx = self.ln(hx)
            if self.quantum_dropout > 0.0:
                hx = self.drop(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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

class LSTMTagger(nn.Module):
    """Sequence tagging model that supports the extended quantum LSTM.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Hidden state size of the LSTM.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of tag classes.
    n_qubits : int, optional
        Number of qubits for the quantum LSTM. If 0, a classical LSTM is used.
    quantum_dropout : float, optional
        Dropout probability applied to the hidden state in the quantum variant.
    layer_norm : bool, optional
        Apply LayerNorm to hidden state if True.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        *,
        quantum_dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                quantum_dropout=quantum_dropout,
                layer_norm=layer_norm,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
