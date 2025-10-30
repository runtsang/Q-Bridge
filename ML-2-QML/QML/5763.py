"""Quantum‑enhanced LSTM tagger with a quantum fully‑connected head.

The architecture mirrors :class:`QLSTMGen` from the classical module
but replaces the LSTM cell and the final projection with quantum
counterparts.  When ``n_qubits`` is set to zero the module falls back
to a pure PyTorch implementation for compatibility.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import encoder_op_list_name_dict


class QLSTMQuantum(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a quantum circuit."""

    class QGate(tq.QuantumModule):
        """Quantum sub‑module implementing a single LSTM gate."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode classical input into qubit rotations
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
            # Entangle wires to mix information
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Gates
        self.forget = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        # Linear maps from classical concatenated state to qubit input
        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.lin_input(combined)))
            g = torch.tanh(self.update(self.lin_update(combined)))
            o = torch.sigmoid(self.output(self.lin_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        out = torch.cat(outputs, dim=0)
        return out, (hx, cx)

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


class QuantumFC(tq.QuantumModule):
    """Quantum fully‑connected head inspired by Quantum‑NAT."""

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class QLSTMGen(tq.QuantumModule):
    """Quantum‑enhanced LSTM tagger.

    The architecture mirrors :class:`QLSTMGen` from the classical module
    but replaces the LSTM cell and the final projection with quantum
    counterparts.  When ``n_qubits`` is set to zero the module falls back
    to a pure PyTorch implementation for compatibility.
    """

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
            self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits)
            self.proj = nn.Linear(hidden_dim, 4).to(device="cpu")  # project to 4‑dim quantum input
            self.fc = QuantumFC()
            self.out_proj = nn.Linear(4, tagset_size)
            self.log_softmax = nn.LogSoftmax(dim=1)
        else:
            # Fallback to pure PyTorch for zero qubits
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, tagset_size)
            self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape ``(batch, seq_len)`` containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape ``(batch, seq_len, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, QLSTMQuantum):
            lstm_out, _ = self.lstm(embeds)
            q_features = self.proj(lstm_out)
            quantum_out = self.fc(q_features)
            logits = self.out_proj(quantum_out)
        else:
            lstm_out, _ = self.lstm(embeds)
            logits = self.fc(lstm_out)
        return self.log_softmax(logits)


__all__ = ["QLSTMGen"]
