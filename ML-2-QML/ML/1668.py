import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

class QLSTM(nn.Module):
    """
    Classic LSTM cell with optional dropout, residual connection, and a switchable gating activation.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0,
                 residual: bool = False,
                 gating: str = "sigmoid") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.residual = residual
        self.gating = gating

        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)

        self.drop = Dropout(p=dropout) if dropout else nn.Identity()

    def _gating(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gating == "sigmoid":
            return torch.sigmoid(logits)
        if self.gating == "softmax":
            return F.softmax(logits, dim=-1)
        raise ValueError(f"Unknown gating type: {self.gating}")

    def forward(self,
                inputs: torch.Tensor,
                states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self._gating(self.forget_lin(combined))
            i = self._gating(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = self._gating(self.output_lin(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.residual:
                hx = hx + x  # residual connection
            hx = self.drop(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple | None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use the extended QLSTM or a standard nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 dropout: float = 0.0,
                 residual: bool = False,
                 gating: str = "sigmoid",
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # Placeholder for quantum variant in the QML implementation
            self.lstm = None
        else:
            self.lstm = QLSTM(embedding_dim,
                              hidden_dim,
                              dropout=dropout,
                              residual=residual,
                              gating=gating)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
