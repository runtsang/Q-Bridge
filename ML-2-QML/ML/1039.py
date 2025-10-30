import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMHybrid(nn.Module):
    """
    Classical LSTM with optional dynamic masking, attention, and a common loss interface.
    The quantum variant is implemented in a separate module but shares the same API.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 depth: int = 1,
                 attention_dim: Optional[int] = None,
                 use_quantum: bool = False,
                 device: torch.device | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.attention_dim = attention_dim
        self.use_quantum = use_quantum

        if device is None:
            device = torch.device('cpu')
        self.device = device

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional attention head
        if attention_dim is not None:
            self.attention = nn.Linear(hidden_dim, attention_dim)
            self.attention_context = nn.Linear(attention_dim, hidden_dim)
        else:
            self.attention = None

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        mask: (seq_len, batch) with 1 for valid tokens, 0 for padding
        states: (h_0, c_0) each (batch, hidden_dim)
        """
        hx, cx = self._init_states(inputs, states)
        seq_len, batch_size, _ = inputs.size()
        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if mask is not None:
                mask_t = mask[t].unsqueeze(1).to(dtype=f.dtype)
                f = f * mask_t + (1 - mask_t) * 1.0
                i = i * mask_t + (1 - mask_t) * 0.0
                g = g * mask_t + (1 - mask_t) * 0.0
                o = o * mask_t + (1 - mask_t) * 1.0

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)

        # Optional attention over hidden states
        if self.attention is not None:
            attn_weights = F.softmax(self.attention(outputs), dim=0)  # (seq_len, batch, attention_dim)
            context = torch.sum(attn_weights * outputs, dim=0)  # (batch, hidden_dim)
            outputs = context.unsqueeze(0).expand(seq_len, batch_size, -1)

        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def compute_loss(self,
                     logits: torch.Tensor,
                     targets: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     lambda_coherence: float = 0.0) -> torch.Tensor:
        """
        Negative log likelihood loss for sequence tagging.
        In the quantum variant, `lambda_coherence` adds a coherence penalty.
        """
        loss = F.nll_loss(logits, targets, reduction='none')
        if mask is not None:
            loss = loss * mask
        loss = loss.mean()

        # No-op coherence penalty in classical mode
        if lambda_coherence > 0.0 and self.use_quantum:
            loss += lambda_coherence * 0.0

        return loss
