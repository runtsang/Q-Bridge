import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MultiHeadAttentionGate(nn.Module):
    """Computes a gate value using multi‑head self‑attention over the concatenated [x, h]."""
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        batch, dim = q.shape
        head_dim = dim // self.n_heads
        q = q.view(batch, self.n_heads, head_dim)
        k = k.view(batch, self.n_heads, head_dim)
        v = v.view(batch, self.n_heads, head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.reshape(batch, dim)
        return torch.sigmoid(self.out(attn_output))

class QLSTM(nn.Module):
    """Classical LSTM cell with attention‑augmented gates and optional residual."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_heads: int = 1,
                 use_residual: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_residual = use_residual
        # Linear layers to produce gate inputs
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Attention gates
        self.forget_gate = MultiHeadAttentionGate(hidden_dim, n_heads)
        self.input_gate = MultiHeadAttentionGate(hidden_dim, n_heads)
        self.update_gate = MultiHeadAttentionGate(hidden_dim, n_heads)
        self.output_gate = MultiHeadAttentionGate(hidden_dim, n_heads)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_gate(self.forget_linear(combined))
            i = self.input_gate(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = self.output_gate(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.use_residual:
                hx = hx + x  # residual from input to hidden
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the enhanced QLSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_heads: int = 1,
                 use_residual: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_heads=n_heads, use_residual=use_residual)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
