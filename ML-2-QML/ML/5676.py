import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQLSTM(nn.Module):
    """Classical LSTM with linear gates and optional dropout."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.dropout = nn.Dropout(dropout)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(self.dropout(hx.unsqueeze(0)))
        return torch.cat(outputs, dim=0), (hx, cx)

class ClassicalSelfAttention(nn.Module):
    """Single‑head self‑attention with linear projections."""
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (seq_len, hidden_dim)
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.dim), dim=-1)
        return self.dropout(scores @ V)

class QLSTMWithAttention(nn.Module):
    """Hybrid LSTM tagger that optionally uses quantum gates and a self‑attention layer."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        attention_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.attention_module = attention_module or ClassicalSelfAttention(hidden_dim, dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence shape: (seq_len, batch_size)
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        # If batch dimension >1, squeeze for attention
        if lstm_out.ndim == 3:
            lstm_out = lstm_out.squeeze(1)
        attn_out = self.attention_module(lstm_out)
        tag_logits = self.hidden2tag(attn_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["QLSTMWithAttention", "ClassicalQLSTM", "ClassicalSelfAttention"]
