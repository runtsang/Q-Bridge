import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Enhanced classical LSTM with optional dropout, bias, and layer normalization."""
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 1,
                 dropout: float = 0.0, bias: bool = True, batch_first: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0.0,
                            bias=bias,
                            batch_first=batch_first)

        # Layer normalization after each LSTM layer
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Forward LSTM
        output, (hn, cn) = self.lstm(inputs, states)
        # Apply layer norm to the output of the last layer (simple approach)
        output = self.norm_layers[-1](output)
        return output, (hn, cn)

    def _init_states(self,
                     batch_size: int,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        hx = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the extended QLSTM or a standard nn.LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 n_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_layers=n_layers,
                              dropout=dropout, bias=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                                batch_first=False,
                                dropout=dropout if n_layers > 1 else 0.0,
                                bias=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
