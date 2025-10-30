"""Quantum‑enhanced LSTM with batched circuits and configurable readout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

import torchquantum as tq
import torchquantum.functional as tqf

class QGateLayer(tq.QuantumModule):
    """Quantum gate that transforms an input vector into a measurement vector.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the circuit.
    readout_wires : List[int], optional
        Which wires to read out.  If ``None`` all wires are measured.
    """

    def __init__(self, n_wires: int, readout_wires: Optional[List[int]] = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.readout_wires = readout_wires or list(range(n_wires))

        # Encode input as RX rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )

        # Parameterized RX gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        # Entanglement via CNOTs
        self.cnot_pattern = [
            (wire, (wire + 1) % n_wires) for wire in range(n_wires)
        ]

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_wires)``.  Values are assumed to be in
            the range ``[-π, π]`` for valid RX rotations.

        Returns
        -------
        torch.Tensor
            Measurement outcomes of shape ``(batch, len(readout_wires))``.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for src, tgt in self.cnot_pattern:
            tqf.cnot(qdev, wires=[src, tgt])
        meas = self.measure(qdev)
        return meas[:, self.readout_wires]

class QLSTM(nn.Module):
    """LSTM where each gate is a small quantum circuit."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to quantum input space
        self.forget_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin   = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate  = QGateLayer(n_qubits)
        self.input_gate   = QGateLayer(n_qubits)
        self.update_gate  = QGateLayer(n_qubits)
        self.output_gate  = QGateLayer(n_qubits)

        # Map quantum measurement back to hidden size
        self.forget_map  = nn.Linear(n_qubits, hidden_dim)
        self.input_map   = nn.Linear(n_qubits, hidden_dim)
        self.update_map  = nn.Linear(n_qubits, hidden_dim)
        self.output_map  = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states : tuple, optional
            Initial hidden and cell states of shape ``(batch, hidden_dim)``.
            If ``None`` zero‑states are used.

        Returns
        -------
        outputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, hidden_dim)``.
        final_state : tuple
            Final hidden and cell states.
        """
        if states is None:
            batch_size = inputs.size(1)
            device = inputs.device
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            h, c = states

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)

            f_q = self.forget_gate(self.forget_lin(combined))
            i_q = self.input_gate(self.input_lin(combined))
            g_q = self.update_gate(self.update_lin(combined))
            o_q = self.output_gate(self.output_lin(combined))

            f = torch.sigmoid(self.forget_map(f_q))
            i = torch.sigmoid(self.input_map(i_q))
            g = torch.tanh(self.update_map(g_q))
            o = torch.sigmoid(self.output_map(o_q))

            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the quantum LSTM."""

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
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
