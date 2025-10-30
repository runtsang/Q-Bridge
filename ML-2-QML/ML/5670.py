import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """
    Lightweight self‑attention module that operates on the concatenated
    input–hidden vector and produces a context vector of the same size.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, embed_dim)
        returns: (batch, embed_dim)
        """
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class QLSTM(nn.Module):
    """
    Classical LSTM cell whose gates are conditioned on a self‑attention
    context computed from the combined input‑hidden vector.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.attention = ClassicalSelfAttention(embed_dim=input_dim + hidden_dim)

        # Linear projections for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            attn_out = self.attention(combined)
            f = torch.sigmoid(self.forget_linear(attn_out))
            i = torch.sigmoid(self.input_linear(attn_out))
            g = torch.tanh(self.update_linear(attn_out))
            o = torch.sigmoid(self.output_linear(attn_out))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the classical QLSTM
    and a vanilla nn.LSTM by setting the ``n_qubits`` flag.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
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
