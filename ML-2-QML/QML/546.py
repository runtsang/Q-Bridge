"""Quantum‑enhanced LSTM module with multi‑layer support and hybrid regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, List, Optional

class _QuantumQLSTMCell(nn.Module):
    """Quantum‑based LSTM cell with a small parameterised circuit for each gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projections into qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum modules for gates
        self.forget_gate = self._make_gate()
        self.input_gate = self._make_gate()
        self.update_gate = self._make_gate()
        self.output_gate = self._make_gate()

    def _make_gate(self) -> tq.QuantumModule:
        """Create a small quantum module that encodes the gate logic."""
        class Gate(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                # Simple encoder mapping classical inputs to rotations
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [i], "func": "rx", "wires": [i]}
                        for i in range(n_wires)
                    ]
                )
                # Parameterised RX gates
                self.params = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(
                    n_wires=self.n_wires,
                    bsz=x.shape[0],
                    device=x.device
                )
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                # Simple entanglement chain
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)

        return Gate(self.n_qubits)

    def _quantum_gate(self, lin_out: torch.Tensor, gate: tq.QuantumModule) -> torch.Tensor:
        """Apply the quantum module and return the expectation as a gate value."""
        return torch.sigmoid(gate(lin_out))

    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = states
        combined = torch.cat([x, h_prev], dim=1)

        f = self._quantum_gate(self.forget_lin(combined), self.forget_gate)
        i = self._quantum_gate(self.input_lin(combined), self.input_gate)
        g = torch.tanh(self._quantum_gate(self.update_lin(combined), self.update_gate))
        o = self._quantum_gate(self.output_lin(combined), self.output_gate)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class QLSTMPlus(nn.Module):
    """Quantum‑enhanced LSTM with optional classical fallback and layer‑wise stacking.

    Parameters
    ----------
    input_dim : int
        Size of each input vector.
    hidden_dim : int
        Size of the hidden state.
    n_layers : int, default=1
        Number of stacked LSTM layers.
    dropout : float, default=0.0
        Dropout probability between layers.
    n_qubits : int, default=4
        Number of qubits used in each quantum gate.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_qubits = n_qubits

        # Build stacked layers
        self.layers: nn.ModuleList = nn.ModuleList()
        for layer in range(n_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(_QuantumQLSTMCell(layer_input_dim, hidden_dim, n_qubits))

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Layer‑norm for each layer
        self.layer_norms: nn.ModuleList = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        seq_len, batch_size, _ = inputs.size()
        if states is None:
            states = [
                (torch.zeros(batch_size, self.hidden_dim, device=inputs.device),
                 torch.zeros(batch_size, self.hidden_dim, device=inputs.device))
                for _ in range(self.n_layers)
            ]

        outputs = []
        new_states: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for t in range(seq_len):
            x = inputs[t]
            layer_input = x
            layer_states = []

            for layer, cell in enumerate(self.layers):
                h, c = states[layer]
                h_next, c_next = cell(layer_input, (h, c))
                h_next = self.layer_norms[layer](h_next)
                layer_input = self.dropout_layer(h_next) if layer < self.n_layers - 1 else h_next
                layer_states.append((h_next, c_next))

            outputs.append(layer_input.unsqueeze(0))
            new_states = layer_states

        outputs = torch.cat(outputs, dim=0)
        return outputs, new_states

    def init_states(self, batch_size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a list of zero hidden states for each layer."""
        return [
            (torch.zeros(batch_size, self.hidden_dim, device=device),
             torch.zeros(batch_size, self.hidden_dim, device=device))
            for _ in range(self.n_layers)
        ]

__all__ = ["QLSTMPlus"]
