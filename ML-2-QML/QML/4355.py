"""Quantum‑enhanced LSTM model that combines a quantum encoder, quantum gates and classical post‑processing."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumEncoder(tq.QuantumModule):
    """Simple quantum encoder that maps a classical vector to a 4‑qubit state via RX gates."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        return self.measure(qdev)

class QuantumQLSTMCell(tq.QuantumModule):
    """Quantum‑LSTM cell that uses a small variational circuit for each gate."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
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
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = states
        combined = torch.cat([inputs, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class HybridQLSTMQuantum(tq.QuantumModule):
    """Hybrid quantum‑classical LSTM that optionally uses a quantum encoder and a quantum‑inspired fraud layer."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_autoencoder: bool = False,
        fraud_clip: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_autoencoder = use_autoencoder

        if use_autoencoder:
            self.encoder = QuantumEncoder()
            self.input_dim = 4  # QuantumEncoder outputs 4 classical bits

        if n_qubits > 0:
            self.lstm_cell = QuantumQLSTMCell(self.input_dim, hidden_dim, n_qubits)
        else:
            # Fallback to a classical LSTM cell for environments without a quantum backend
            self.lstm_cell = nn.LSTMCell(self.input_dim, hidden_dim)

        # Classical fraud‑style regulariser applied after the quantum gates
        self.fraud_layer = nn.Linear(2, 2)
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).

        Returns
        -------
        out : torch.Tensor
            Predictions of shape (seq_len, batch, 1).
        states : Tuple[torch.Tensor, torch.Tensor]
            Final hidden and cell states.
        """
        hx = torch.zeros(inputs.size(1), self.hidden_dim, device=inputs.device)
        cx = torch.zeros(inputs.size(1), self.hidden_dim, device=inputs.device)
        outputs = []

        for t in range(inputs.size(0)):
            x = inputs[t]
            if self.use_autoencoder:
                x = self.encoder(x.unsqueeze(0)).squeeze(0)
            hx, cx = self.lstm_cell(x, (hx, cx))
            hx = self.fraud_layer(hx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        out = self.output_linear(outputs)
        return out, (hx, cx)

class LSTMTaggerQuantum(tq.QuantumModule):
    """Sequence tagging model that uses :class:`HybridQLSTMQuantum`."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_autoencoder: bool = False,
        fraud_clip: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTMQuantum(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_autoencoder=use_autoencoder,
            fraud_clip=fraud_clip,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTMQuantum", "LSTMTaggerQuantum", "QuantumEncoder", "QuantumQLSTMCell"]
