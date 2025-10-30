import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen025(nn.Module):
    """
    Classical LSTM cell with dropout and multi‑task heads.
    Extends the original QLSTM interface by:

    * Supporting variable sequence lengths via pack_padded_sequence.
    * Optional dropout after the recurrent step.
    * A list of head modules to produce logits for multiple tasks
      (e.g. POS tagging and chunking).
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_tasks: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        self.dropout = dropout

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.task_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                         for _ in range(n_tasks)])

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a batch of sequences.

        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            seq_lengths: Optional tensor of shape (batch,) with actual lengths.

        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            final_states: Tuple (hx, cx) each of shape (batch, hidden_dim)
        """
        hx, cx = self._init_states(inputs)

        if seq_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(inputs, seq_lengths.cpu(),
                                                       enforce_sorted=False)
        else:
            packed = inputs

        outputs = []
        if isinstance(packed, torch.Tensor):
            iterable = packed.unbind(dim=0)
        else:
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed)
            iterable = unpacked.unbind(dim=0)

        for x in iterable:
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            hx = self.dropout_layer(hx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def get_task_outputs(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Compute outputs for all tasks.

        Args:
            lstm_out: Tensor of shape (seq_len, batch, hidden_dim)
        Returns:
            Tensor of shape (seq_len, batch, n_tasks, hidden_dim)
        """
        return torch.stack([head(lstm_out) for head in self.task_heads], dim=2)

    def log_gate_counts(self) -> None:
        """
        No quantum gates in classical version; method is provided for API parity.
        """
        pass

__all__ = ["QLSTMGen025"]
