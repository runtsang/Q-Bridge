"""Quantum‑enhanced LSTM cell for fraud detection.

This module implements a QLSTM that replaces the classical gate
computations by variational quantum circuits.  It is designed to
be imported by the classical hybrid model defined in the ML module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """LSTM cell with quantum‑realised gates.

    The four gates (forget, input, update, output) are each computed
    by a small 4‑qubit variational circuit.  The circuit is built
    from a parameterised RX encoder followed by a chain of CNOTs.
    """

    class _QLayer(tq.QuantumModule):
        """Quantum sub‑module that implements a single LSTM gate."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encoder: each input dimension is rotated around X
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_wires)
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=[i])
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

        # Linear projections to the qubit space
        self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_proj(combined)))
            i = torch.sigmoid(self.input_gate(self.input_proj(combined)))
            g = torch.tanh(self.update_gate(self.update_proj(combined)))
            o = torch.sigmoid(self.output_gate(self.output_proj(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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

__all__ = ["QLSTM"]
