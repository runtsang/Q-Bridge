import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(QLSTM):
    """
    A hybrid classical‑quantum LSTM that augments the original QLSTM
    with a multi‑head attention mechanism and a learnable readout
    to combine the hidden states across time.  The attention
    weights are produced by a small feed‑forward network that
    operates on the concatenated hidden and cell states.
    The class remains compatible with the original API: it
    provides ``forward`` and ``_init_states`` methods and
    can directly replace the ``QLSTM`` class in the tagger.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        num_heads: int = 2,
        readout_dim: int = 30,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(input_dim, hidden_dim, n_qubits)
        self.num_heads = num_heads
        self.readout_dim = readout_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )
        self.readout = nn.Linear(num_heads * hidden_dim, readout_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        hidden_states = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))
            hidden_states.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        hidden_cat = torch.cat(hidden_states, dim=0)

        # Compute attention weights
        attn_scores = self.attention(torch.cat([hidden_cat, cx.unsqueeze(0).repeat(hidden_cat.size(0), 1)], dim=1))
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum of hidden states
        weighted_hidden = torch.sum(attn_weights.unsqueeze(-1) * hidden_cat, dim=0)
        readout = self.readout(weighted_hidden)

        return stacked, (hx, cx)
