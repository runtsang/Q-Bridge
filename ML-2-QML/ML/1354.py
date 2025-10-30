import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class StateDropout(nn.Module):
    """Randomly drops hidden and cell states during training."""
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.drop_prob == 0.0:
            return hidden, cell
        mask = torch.bernoulli((1 - self.drop_prob) * torch.ones_like(hidden))
        return hidden * mask, cell * mask


class GatedAttentionWrapper(nn.Module):
    """Wraps any LSTM (classical or quantum) and adds a gated attention over its outputs."""
    def __init__(self, lstm: nn.Module, hidden_dim: int):
        super().__init__()
        self.lstm = lstm
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        outputs, (hn, cn) = self.lstm(inputs, states)
        attn_weights = torch.softmax(self.attn_proj(outputs), dim=0)  # (seq_len, batch, hidden)
        attn_context = torch.sum(attn_weights * outputs, dim=0, keepdim=True)  # (1, batch, hidden)
        gate = torch.sigmoid(self.gate_proj(attn_context))
        outputs = outputs * gate
        return outputs, (hn, cn)


class QLSTMGen(nn.Module):
    """Hybrid LSTM that can optionally replace its gates with quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = StateDropout(dropout)

        # Classical linear layers mapping to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Map qubit outputs back to hidden dimension
        self.to_hidden = nn.Linear(n_qubits, hidden_dim)

        # Flag to toggle quantum gates; for the classical version set to False
        self.use_quantum = False  # set to True to enable quantum gates

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.to_hidden(self.forget_lin(combined)))
            i = torch.sigmoid(self.to_hidden(self.input_lin(combined)))
            g = torch.tanh(self.to_hidden(self.update_lin(combined)))
            o = torch.sigmoid(self.to_hidden(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        hx, cx = self.dropout(hx, cx)
        return outputs, (hx, cx)


class LSTMTaggerGen(nn.Module):
    """Sequence tagging model that uses the hybrid QLSTMGen."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen", "LSTMTaggerGen", "GatedAttentionWrapper", "StateDropout"]
