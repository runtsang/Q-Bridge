"""QuanvolutionQLSTMHybrid – quantum‑centric implementation.

This module implements the same high‑level API as the classical
counterpart but replaces the convolutional filter and the LSTM
with quantum‑enhanced components.  The quantum filter applies
a random 2‑qubit circuit to each 2×2 image patch, while the
QLSTM uses small parametric circuits for the gates.  Both
modules are fully trainable via PyTorch’s autograd.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["QuanvolutionFilter", "QLSTM", "QuanvolutionQLSTMHybrid"]


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum patch encoder that applies a random two‑qubit circuit to each 2×2 patch."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
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
        # Stack patches: shape (batch, 196, 4)
        return torch.stack(patches, dim=1)


class QLSTM(tq.QuantumModule):
    """Quantum LSTM where each gate is a small parametrised quantum circuit."""

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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QuanvolutionQLSTMHybrid(tq.QuantumModule):
    """Hybrid model that combines a quantum filter, a quantum LSTM,
    and a linear classifier.  It can fall back to classical
    implementations by setting `use_qfilter` or `use_qlstm` to False."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 10,
        n_qubits_filter: int = 4,
        n_qubits_lstm: int = 4,
        use_qfilter: bool = True,
        use_qlstm: bool = True,
    ) -> None:
        super().__init__()
        self.use_qfilter = use_qfilter
        self.use_qlstm = use_qlstm

        if use_qfilter:
            self.filter = QuanvolutionFilter(n_wires=n_qubits_filter)
        else:
            # Classical conv fallback
            self.filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        if use_qlstm:
            self.lstm = QLSTM(input_dim=4, hidden_dim=hidden_dim, n_qubits=n_qubits_lstm)
        else:
            # Classical LSTM fallback
            self.lstm = nn.LSTM(input_dim=4, hidden_dim=hidden_dim, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Filter
        if self.use_qfilter:
            patches = self.filter(x)  # (batch, 196, 4)
        else:
            conv_out = self.filter(x)  # (batch, 4, 14, 14)
            patches = conv_out.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)  # (batch, 196, 4)

        # LSTM
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(patches)  # (batch, 196, hidden_dim)
        else:
            # QLSTM expects input of shape (seq_len, batch, input_dim)
            seq = patches.permute(1, 0, 2)  # (seq_len, batch, 4)
            lstm_out, _ = self.lstm(seq)  # (seq_len, batch, hidden_dim)
            lstm_out = lstm_out.permute(1, 0, 2)  # (batch, seq_len, hidden_dim)

        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        logits = self.classifier(last_hidden)
        return F.log_softmax(logits, dim=-1)
