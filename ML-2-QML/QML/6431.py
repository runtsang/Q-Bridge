"""Hybrid quantum‑classical module that uses a quantum quanvolution filter
and a quantum‑augmented LSTM head.

Author: OpenAI GPT‑OSS‑20B
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

__all__ = ["HybridQuanvolutionQLSTM"]

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a parametric 2‑qubit kernel to each 2×2 patch.

    The implementation follows the spirit of the original `QuanvolutionFilter`
    but replaces the fixed convolution with a small variational circuit.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each patch into the qubits using Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Sequence of quantum measurements with shape (B, seq_len, n_wires).
        """
        bsz, _, h, w = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, self.n_wires))
        seq = torch.cat(patches, dim=1).view(bsz, (h // 2) * (w // 2), self.n_wires)
        return seq

class QuantumQLSTM(tq.QuantumModule):
    """LSTM where each gate is realised by a small quantum circuit."""
    class QGate(tq.QuantumModule):
        """Single quantum gate that maps a classical vector to a qubit state."""
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
                target = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, target])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, seq: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        seq : torch.Tensor
            Input sequence of shape (B, seq_len, input_dim).
        states : Tuple[torch.Tensor, torch.Tensor] | None
            Optional initial hidden and cell states.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The sequence of hidden states and the final (hidden, cell) tuple.
        """
        batch_size = seq.shape[0]
        seq_len = seq.shape[1]
        hx, cx = self._init_states(seq, states)
        outputs = []
        for t in range(seq_len):
            x_t = seq[:, t, :]
            combined = torch.cat([x_t, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)

    def _init_states(self, seq: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = seq.shape[0]
        device = seq.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridQuanvolutionQLSTM(tq.QuantumModule):
    """Full quantum‑classical pipeline: quantum filter → quantum LSTM → classifier."""
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10, n_qubits: int = 4) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter(n_wires=n_qubits)
        self.lstm = QuantumQLSTM(input_dim=n_qubits, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, num_classes).
        """
        seq = self.filter(x)  # (B, seq_len, n_qubits)
        outputs, (hx, _) = self.lstm(seq)
        logits = self.classifier(hx)
        return F.log_softmax(logits, dim=-1)
