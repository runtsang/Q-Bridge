"""Extended classical LSTM module with optional quantum gate back‑bones and stacking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class _LinearQLSTMCell(nn.Module):
    """Classical linear‑gated LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = states
        combined = torch.cat([x, h_prev], dim=1)
        f = torch.sigmoid(self.forget(combined))
        i = torch.sigmoid(self.input_gate(combined))
        g = torch.tanh(self.update(combined))
        o = torch.sigmoid(self.output(combined))
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class _QuantumQLSTMCell(nn.Module):
    """Quantum‑inspired LSTM cell using a tiny parameterised circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projections to quantum space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum module placeholder (simple RX gates)
        self.register_parameter('quantum_weights', nn.Parameter(torch.randn(n_qubits, 3)))

    def _quantum_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a toy quantum circuit to each sample in the batch.
        The circuit consists of RX rotations followed by a simple entangling pattern.
        """
        # Simple simulation: treat x as rotation angles
        # For demonstration, we just return the same shape
        return torch.sin(x)  # placeholder for actual quantum output

    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = states
        combined = torch.cat([x, h_prev], dim=1)

        f = torch.sigmoid(self._quantum_circuit(self.forget_lin(combined)))
        i = torch.sigmoid(self._quantum_circuit(self.input_lin(combined)))
        g = torch.tanh(self._quantum_circuit(self.update_lin(combined)))
        o = torch.sigmoid(self._quantum_circuit(self.output_lin(combined)))

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class QLSTMPlus(nn.Module):
    """A flexible LSTM cell that can use either linear or quantum gates and supports multi‑layer stacking.

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
    gate_type : str, default='linear'
        ``'linear'`` for classical gates, ``'quantum'`` for quantum‑inspired gates.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        gate_type: str = 'linear',
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.gate_type = gate_type

        # Build stacked layers
        self.layers: nn.ModuleList = nn.ModuleList()
        for layer in range(n_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            if gate_type == 'linear':
                layer_module = _LinearQLSTMCell(layer_input_dim, hidden_dim)
            else:
                layer_module = _QuantumQLSTMCell(layer_input_dim, hidden_dim)
            self.layers.append(layer_module)

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
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (seq_len, batch, input_dim).
        states : list of tuples
            Optional list of (h, c) for each layer.

        Returns
        -------
        outputs : torch.Tensor
            Sequence of shape (seq_len, batch, hidden_dim) from the last layer.
        new_states : list of tuples
            Updated hidden states for each layer.
        """
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
