"""Hybrid model: classical quanvolution filter followed by quantum LSTM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class ClassicalQuanvolutionFilter(nn.Module):
    """Same as in classical version; kept for consistency."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.conv(x)
        return self.flatten(features)

class QLSTM(nn.Module):
    """Quantum LSTM cell where each gate is implemented by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.rxs = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                      for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate, wire in zip(self.rxs, range(self.n_wires)):
                gate(qdev, wires=wire)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])  # wrap‑around CNOT
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = None, lstm_layers: int = 1) -> None:
        super().__init__()
        if n_qubits is None:
            n_qubits = hidden_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.lstm_layers = lstm_layers

        # Linear projections to quantum gate input size
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum layers representing gates
        self.forget_gate = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update_gate = self.QLayer(n_qubits)
        self.output_gate = self.QLayer(n_qubits)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class QuanvolutionQLSTM(nn.Module):
    """Hybrid model: classical quanvolution filter + quantum LSTM."""
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10,
                 lstm_layers: int = 1, n_qubits: int = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        seq_len = 14 * 14
        feature_dim = 4
        self.lstm = QLSTM(input_dim=feature_dim,
                          hidden_dim=hidden_dim,
                          n_qubits=n_qubits,
                          lstm_layers=lstm_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        flat = self.filter(x)                    # [batch, seq_len*feature_dim]
        seq = flat.view(batch_size, 14 * 14, 4)   # [batch, seq_len, feature_dim]
        lstm_out, _ = self.lstm(seq.unsqueeze(1))  # [seq_len, batch, hidden_dim]
        logits = self.classifier(lstm_out[-1])     # take last time step
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionQLSTM"]
