"""Quantum‑enhanced LSTM cell used by the hybrid GraphQNN_LSTM model.

The implementation is a direct rewrite of the classical QLSTM from the
seed code, but expressed with a clean, self‑contained module that can be
imported independently.  It relies on the `torchquantum` library for
variational circuits and measurement.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QLSTM(nn.Module):
    """Hybrid LSTM cell where each gate is a small variational quantum circuit."""

    class _QLayer(tq.QuantumModule):
        """Gate‑specific quantum circuit used in the QLSTM."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encoder: linear rotation of each input qubit
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Parameterised gates: one RX per qubit
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=x.shape[0],
                device=x.device,
            )
            self.encoder(qdev, x)
            for idx, gate in enumerate(self.params):
                gate(qdev, wires=idx)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: list[torch.Tensor] = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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
