import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMPlus(nn.Module):
    """
    Classical LSTM with enhanced regularisation and a gated‑attention read‑out.
    The class keeps the original QLSTM API so it can be swapped in without
    changing downstream code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual

        # Classical gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_layernorm:
            self.ln_forget = nn.LayerNorm(hidden_dim)
            self.ln_input = nn.LayerNorm(hidden_dim)
            self.ln_update = nn.LayerNorm(hidden_dim)
            self.ln_output = nn.LayerNorm(hidden_dim)

        # Attention read‑out
        self.attn = nn.Linear(hidden_dim, 1)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        returns: (seq_len, batch, hidden_dim), (hx, cx)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.use_layernorm:
                f = self.ln_forget(f)
                i = self.ln_input(i)
                g = self.ln_update(g)
                o = self.ln_output(o)

            f, i, g, o = map(self.dropout, (f, i, g, o))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            if self.use_residual:
                hx = hx + combined[:, :self.hidden_dim]

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)

        # Gated‑attention read‑out over the whole sequence
        attn_weights = F.softmax(self.attn(outputs).squeeze(-1), dim=0)  # (seq_len,)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=0)  # (batch, hidden_dim)
        # The context vector can be used downstream; here we simply ignore it to preserve API

        return outputs, (hx, cx)

__all__ = ["QLSTMPlus"]
