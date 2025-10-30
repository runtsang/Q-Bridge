"""Quantum implementation of the same hybrid model using a quantum QuanvolutionFilter
and a quantum LSTM layer. The quantum LSTM gates are realized as variational circuits,
and the patch encoder applies a random two‑qubit kernel to each 2×2 image patch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        bsz, device = x.shape[0], x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        seq = torch.cat(patches, dim=1).view(bsz, -1, 4)  # (batch, seq_len, 4)
        return seq


class QLSTM(tq.QuantumModule):
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
            qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, hidden_dim: int, n_layers: int = 1, n_qubits: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_qubits = n_qubits

        # Quantum layers for each LSTM gate
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections from (4 + hidden_dim) → n_qubits
        self.lin_forget = nn.Linear(4 + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(4 + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(4 + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(4 + hidden_dim, n_qubits)

        # Map qubit measurements to hidden_dim
        self.measure_to_hidden = nn.Linear(n_qubits, hidden_dim)

    def forward(self, seq: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # seq: (batch, seq_len, 4)
        batch, seq_len, _ = seq.shape
        hx, cx = self._init_states(seq, states)
        outputs = []
        for t in range(seq_len):
            x_t = seq[:, t, :]  # (batch, 4)
            combined = torch.cat([x_t, hx], dim=1)  # (batch, 4 + hidden_dim)
            f = torch.sigmoid(self.measure_to_hidden(self.forget(self.lin_forget(combined))))
            i = torch.sigmoid(self.measure_to_hidden(self.input(self.lin_input(combined))))
            g = torch.tanh(self.measure_to_hidden(self.update(self.lin_update(combined))))
            o = torch.sigmoid(self.measure_to_hidden(self.output(self.lin_output(combined))))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        lstm_out = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return lstm_out, (hx, cx)

    def _init_states(self,
                     seq: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = seq.shape[0]
        device = seq.device
        hx = torch.zeros(batch, self.hidden_dim, device=device)
        cx = torch.zeros(batch, self.hidden_dim, device=device)
        return hx, cx


class QLSTMQuanvolutionClassifier(nn.Module):
    """
    Quantum hybrid model that uses a quantum QuanvolutionFilter to
    extract 2×2 patches, then processes the resulting sequence with a
    quantum LSTM layer, followed by a classical linear classifier.
    """
    def __init__(self,
                 hidden_dim: int = 128,
                 n_layers: int = 1,
                 n_qubits: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.lstm = QLSTM(hidden_dim=hidden_dim,
                          n_layers=n_layers,
                          n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.qfilter(x)          # (batch, seq_len, 4)
        lstm_out, _ = self.lstm(seq)   # (batch, seq_len, hidden_dim)
        logits = self.classifier(lstm_out[:, -1, :])  # last hidden state
        return F.log_softmax(logits, dim=-1)


__all__ = ["QLSTMQuanvolutionClassifier"]
