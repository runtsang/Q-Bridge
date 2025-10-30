import torch
import torch.nn as nn
from typing import Tuple

class QLSTM(nn.Module):
    """Drop‑in classical LSTM replacement with optional regularisation and gate freezing."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        freeze_gates: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # retained for API compatibility
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        self.freeze_gates = freeze_gates

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.use_layer_norm:
            self.forget_ln = nn.LayerNorm(gate_dim)
            self.input_ln = nn.LayerNorm(gate_dim)
            self.update_ln = nn.LayerNorm(gate_dim)
            self.output_ln = nn.LayerNorm(gate_dim)

        if self.freeze_gates:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.use_layer_norm:
                f = self.forget_ln(f)
                i = self.input_ln(i)
                g = self.update_ln(g)
                o = self.output_ln(o)

            f = self.dropout(f)
            i = self.dropout(i)
            g = self.dropout(g)
            o = self.dropout(o)

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
