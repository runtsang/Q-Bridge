import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class ClassicalSelfAttention(nn.Module):
    """Differentiable dot‑product self‑attention with learnable parameters."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable linear maps for query/key/value
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (seq_len, batch, embed_dim)
        seq_len, batch, embed_dim = inputs.shape
        # Permute to (batch, seq_len, embed_dim) for batch-wise attention
        inp = inputs.permute(1, 0, 2)
        query = torch.einsum('bte,ef->btf', inp, self.rotation_params)
        key   = torch.einsum('bte,ef->btf', inp, self.entangle_params)
        scores = torch.softmax(torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.embed_dim), dim=-1)
        context = torch.bmm(scores, inp)
        return context.permute(1, 0, 2)  # back to (seq_len, batch, embed_dim)

class QLSTM(nn.Module):
    """Hybrid LSTM cell that optionally uses self‑attention."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical LSTM gates
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Self‑attention
        self.attention = ClassicalSelfAttention(embed_dim=input_dim)
        self.context_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        # Pre‑compute attention for the entire sequence
        context = self.attention(inputs)  # (seq_len, batch, input_dim)

        for idx, x in enumerate(inputs.unbind(dim=0)):
            ctx = context[idx]          # (batch, input_dim)
            ctx_proj = self.context_proj(ctx)  # (batch, hidden_dim)
            hx_proj = hx + ctx_proj
            combined = torch.cat([x, hx_proj], dim=1)  # (batch, input_dim + hidden_dim)

            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
