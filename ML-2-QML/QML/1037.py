"""Quantum‑enhanced LSTM layers for sequence tagging with noise and dropout.

The implementation extends the seed by adding:
* a configurable depolarizing noise channel applied to each qubit,
* an optional dropout on the quantum gate outputs,
* a multi‑layer variational circuit that can be tuned with the number of qubits.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Variational quantum layer that outputs a single real value.

    The circuit consists of:
    * an encoding of the classical input via RX gates,
    * a trainable RX layer,
    * a CNOT ladder to entangle the qubits,
    * measurement of all qubits followed by a linear readout.
    """

    def __init__(self, n_wires: int, dropout: float = 0.0, noise_level: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dropout = dropout
        self.noise_level = noise_level

        # Encode classical features into qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )

        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        # Entanglement
        self.cnot_layers = nn.ModuleList(
            [
                tq.CNOT(wires=[i, (i + 1) % n_wires])
                for i in range(n_wires)
            ]
        )

        # Measurement and readout
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.readout = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x has shape (batch, n_wires)."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)

        # Encoding
        self.encoder(qdev, x)

        # Trainable layers
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)

        # Entanglement
        for gate in self.cnot_layers:
            gate(qdev)

        # Optional depolarizing noise during training
        if self.training and self.noise_level > 0.0:
            for wire in range(self.n_wires):
                tqf.depolarizing_noise(qdev, wires=[wire], prob=self.noise_level)

        # Measurement
        meas = self.measure(qdev)  # shape (batch, n_wires)
        out = self.readout(meas)  # shape (batch, 1)
        out = torch.sigmoid(out).squeeze(-1)  # shape (batch,)
        # Optional dropout on the output
        if self.training and self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout, training=True)
        return out


class QLSTMEnhanced(nn.Module):
    """Quantum‑enhanced LSTM cell that can be used inside a tagger."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout
        self.noise_level = noise_level

        # Quantum layers for each gate
        self.forget = QLayer(n_qubits, dropout=dropout, noise_level=noise_level)
        self.input = QLayer(n_qubits, dropout=dropout, noise_level=noise_level)
        self.update = QLayer(n_qubits, dropout=dropout, noise_level=noise_level)
        self.output = QLayer(n_qubits, dropout=dropout, noise_level=noise_level)

        # Classical linear projections to the quantum workspace
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
                noise_level=noise_level,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMEnhanced", "LSTMTagger"]
