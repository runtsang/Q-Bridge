from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """
    Lightweight feed‑forward sampler that maps the hidden state into four gate logits.
    The network is intentionally shallow to keep training fast while providing
    expressive gating signals.
    """
    def __init__(self, input_dim: int, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

class SamplerQLSTM(nn.Module):
    """
    Classical hybrid sampler‑LSTM.
    The sampler generates four gating probabilities that directly control the
    LSTM cell.  The hidden state is updated exactly as in a standard LSTM but
    with the gates supplied by the sampler network.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sampler = SamplerModule(hidden_dim, output_dim=4)
        self.n_qubits = n_qubits

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            x_proj = self.input_proj(x)
            hx_proj = self.hidden_proj(hx)
            combined = x_proj + hx_proj
            gate_probs = self.sampler(combined)  # [batch, 4]
            f, i, g, o = gate_probs.split(1, dim=-1)
            cx = f * cx + i * torch.tanh(g)
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

__all__ = ["SamplerQLSTM"]
