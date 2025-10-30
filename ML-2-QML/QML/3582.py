"""Hybrid image‑sequence classifier: quantum quanvolution + quantum LSTM.

This module implements the same interface as the classical variant
but replaces the convolutional front‑end and the LSTM gates with
small quantum circuits.  The quanvolution filter applies a random
two‑qubit circuit to each 2×2 image patch; the output of each patch
is a 4‑dimensional measurement vector.  The QLSTM uses a quantum
circuit per gate (forget, input, update, output) and combines the
result with the classical linear projections.  The model is fully
compatible with PyTorch autograd and can be trained on a CPU or
GPU that supports the TorchQuantum backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["QuanvolutionFilter", "QLSTM", "QuanvolutionQLSTM"]


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution: 2×2 patch encoder + random quantum layer."""

    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
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
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (B, seq_len, feat_dim)."""
        bsz = x.shape[0]
        device = x.device
        # flatten to (B, 28, 28)
        img = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        img[:, r, c],
                        img[:, r, c + 1],
                        img[:, r + 1, c],
                        img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, -1))
        seq = torch.stack(patches, dim=1)  # (B, seq_len, 4)
        return seq


class QLSTM(nn.Module):
    """Quantum LSTM where each gate is a small quantum circuit."""

    class _QLayer(tq.QuantumModule):
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
            # Trainable RX gates per wire
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle wires in a chain
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = n_qubits  # hidden dimension matches qubit count
        self.n_qubits = n_qubits
        # Quantum gates
        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)
        # Classical linear projections to match qubit count
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class QuanvolutionQLSTM(nn.Module):
    """Hybrid classifier: quantum quanvolution → quantum LSTM → linear head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    hidden_dim : int, optional
        Hidden dimension of the LSTM. Defaults to 128.
    n_qubits : int, optional
        Number of qubits per quantum gate. Defaults to 4.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 128, n_qubits: int = 4) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(n_wires=n_qubits)
        self.lstm = QLSTM(input_dim=4, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.filter(x)  # (B, seq_len, 4)
        seq = seq.permute(1, 0, 2)  # (seq_len, B, 4)
        lstm_out, _ = self.lstm(seq)
        final_hidden = lstm_out[-1]
        logits = self.classifier(final_hidden)
        return F.log_softmax(logits, dim=-1)
