"""Quantum hybrid model combining a quanvolution filter and a quantum LSTM for sequence classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum 2×2 convolutional filter that processes image patches."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder maps each patch pixel to a qubit rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Parameterized ansatz
        self.ansatz = tq.Sequential(
            tq.RX(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
            tq.RZ(has_params=True, trainable=True),
            tq.CNOT(wires=[0, 1]),
            tq.CNOT(wires=[1, 2]),
            tq.CNOT(wires=[2, 3]),
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels=1, height=28, width=28)
        batch = x.shape[0]
        device = x.device
        # Extract 2x2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 28, 28, 2, 2)
        patches = patches.contiguous().view(batch, 14*14, 4)  # (batch, seq_len, 4)
        seq_len = patches.shape[1]
        # Prepare quantum device
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch*seq_len, device=device)
        self.encoder(qdev, patches.view(-1, self.n_wires))
        self.ansatz(qdev)
        measurement = self.measure(qdev)
        return measurement.view(batch, seq_len, self.n_wires)

class QuantumQLSTM(tq.QuantumModule):
    """Quantum LSTM cell where each gate is implemented by a small quantum circuit."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            self.ansatz = tq.Sequential(
                tq.RX(has_params=True, trainable=True),
                tq.RY(has_params=True, trainable=True),
                tq.RZ(has_params=True, trainable=True),
                tq.CNOT(wires=[0, 1]),
                tq.CNOT(wires=[1, 2]),
                tq.CNOT(wires=[2, 3]),
            )
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=device)
            self.encoder(qdev, x)
            self.ansatz(qdev)
            return self.measure(qdev)
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Linear projections to n_qubits
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        # Linear mappings from quantum gates to hidden state dimension
        self.f_gate = nn.Linear(n_qubits, hidden_dim)
        self.i_gate = nn.Linear(n_qubits, hidden_dim)
        self.g_gate = nn.Linear(n_qubits, hidden_dim)
        self.o_gate = nn.Linear(n_qubits, hidden_dim)
        # Quantum gates
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)
    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch, seq_len, _ = inputs.shape
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            combined = torch.cat([x_t, hx], dim=1)
            f = torch.sigmoid(self.f_gate(self.forget_gate(self.linear_forget(combined))))
            i = torch.sigmoid(self.i_gate(self.input_gate(self.linear_input(combined))))
            g = torch.tanh(self.g_gate(self.update_gate(self.linear_update(combined))))
            o = torch.sigmoid(self.o_gate(self.output_gate(self.linear_output(combined))))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)
        return out_seq, (hx, cx)
    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch = inputs.size(0)
        device = inputs.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

class QuanvolutionLSTMTagger(tq.QuantumModule):
    """
    Hybrid quantum model that applies a quantum quanvolution filter to extract
    local patches, then processes the resulting sequence with a quantum LSTM.
    """
    def __init__(self, hidden_dim: int, num_classes: int, n_qubits: int = 4):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(n_wires=n_qubits)
        self.lstm = QuantumQLSTM(input_dim=n_qubits, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels=1, height=28, width=28)
        Returns:
            log probabilities over classes.
        """
        # Extract patches sequence
        seq = self.qfilter(x)  # shape: (batch, seq_len=196, n_qubits)
        lstm_out, _ = self.lstm(seq)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumQuanvolutionFilter", "QuantumQLSTM", "QuanvolutionLSTMTagger"]
