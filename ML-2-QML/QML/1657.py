"""Quantum‑enhanced LSTM with multi‑layer support and dropout."""

from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(nn.Module):
    """Quantum sub‑module that maps a vector to a qubit measurement."""

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that applies RX gates parameterised by the input
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class QLSTMCell(nn.Module):
    """Single quantum‑enhanced LSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        cx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM that mirrors the public API of the original
    ``QLSTM`` class.  It supports multiple layers, dropout and returns
    the same tuple of tensors as ``torch.nn.LSTM``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(QLSTMCell(layer_input_dim, hidden_dim, n_qubits))

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[List[torch.Tensor], List[torch.Tensor]] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states : tuple or None
            Initial hidden and cell states.  Each is a list of tensors
            of shape ``(batch, hidden_dim)`` per layer.  If ``None`` zero
            states are used.
        """
        seq_len, batch, _ = inputs.size()
        if states is None:
            h, c = [], []
            for _ in range(self.num_layers):
                h.append(torch.zeros(batch, self.hidden_dim, device=inputs.device))
                c.append(torch.zeros(batch, self.hidden_dim, device=inputs.device))
        else:
            h, c = states

        outputs = []
        for t in range(seq_len):
            x_t = inputs[t]
            for l, layer in enumerate(self.layers):
                h_l, c_l = layer(x_t, h[l], c[l])
                h[l], c[l] = h_l, c_l
                x_t = h_l
                if self.dropout and l!= self.num_layers - 1:
                    x_t = self.dropout(x_t)
            outputs.append(h[-1].unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        return outputs, (h, c)


__all__ = ["QLSTM"]
