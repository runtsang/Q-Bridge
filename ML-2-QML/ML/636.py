import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, MultiheadAttention

class HybridGate(nn.Module):
    """Feed‑forward gate with dropout and layer‑norm."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        out = self.linear(combined)
        out = self.dropout(out)
        out = self.norm(out)
        return out

class QLSTM(nn.Module):
    """
    Classical LSTM with quantum‑inspired hybrid gates and multi‑head attention.
    Gates are computed by a small feed‑forward network followed by dropout and
    layer‑norm.  The hidden state is enriched with a multi‑head attention
    over all previously produced hidden states, providing a long‑range context.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_gate = HybridGate(input_dim, hidden_dim, dropout)
        self.input_gate = HybridGate(input_dim, hidden_dim, dropout)
        self.update_gate = HybridGate(input_dim, hidden_dim, dropout)
        self.output_gate = HybridGate(input_dim, hidden_dim, dropout)

        self.attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional tuple of previous (hx, cx)
        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            (hx, cx): Final hidden and cell states
        """
        batch_size = inputs.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        if states is not None:
            hx, cx = states

        outputs = []
        for t in range(inputs.size(0)):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Attention over past hidden states
            if outputs:
                past = torch.stack(outputs, dim=1)  # (batch, t, hidden)
                attn_output, _ = self.attention(
                    query=hx.unsqueeze(1),
                    key=past,
                    value=past,
                )
                hx = hx + attn_output.squeeze(1)

            outputs.append(hx)

        stacked = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the hybrid QLSTM
    and a baseline nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_heads=n_heads, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: Tensor of shape (seq_len, batch)
        Returns:
            log probabilities over tags: shape (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
