import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional dropout and weight normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        proj_dim = hidden_dim
        self.forget = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))
        self.input = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))
        self.update = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))
        self.output = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self,
                inputs: torch.Tensor,
                states: tuple | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
            outputs.append(self.dropout(hx.unsqueeze(0)))
        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the enhanced classical LSTM
    and the built‑in nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 dropout: float = 0.0,
                 use_qlstm: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_qlstm:
            self.lstm = QLSTM(embedding_dim, hidden_dim, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # (1, seq_len, embed_dim)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds.squeeze(0))
        attn_weights = torch.softmax(torch.sum(lstm_out, dim=2), dim=1).unsqueeze(2)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        tag_logits = self.hidden2tag(context)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
