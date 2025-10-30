import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTMGen384(nn.Module):
    """Classical LSTM cell with dropout and a hybrid sigmoid head."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, shift: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        gate_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        logits = self.classifier(stacked)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

__all__ = ["QLSTMGen384"]
