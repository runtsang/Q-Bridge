import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(nn.Module):
    """
    Classical LSTM with optional depth. The API mirrors the quantum version for compatibility.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 depth: int = 1, use_hybrid: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.use_hybrid = use_hybrid
        # Stack of LSTM layers to support depth
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=depth,
                            batch_first=True)
        # Optional linear projection after lstm
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        inputs: Tensor of shape (batch, seq_len, input_dim)
        """
        if states is None:
            h0 = torch.zeros(self.depth, inputs.size(0), self.hidden_dim,
                             device=inputs.device)
            c0 = torch.zeros(self.depth, inputs.size(0), self.hidden_dim,
                             device=inputs.device)
        else:
            h0, c0 = states
        out, (hn, cn) = self.lstm(inputs, (h0, c0))
        out = self.proj(out)
        return out, (hn, cn)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the classical QLSTMGen.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1,
                 use_hybrid: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits=n_qubits,
                             depth=depth, use_hybrid=use_hybrid)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.LongTensor) -> torch.Tensor:
        """
        sentence: LongTensor of shape (batch, seq_len)
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["QLSTMGen", "LSTMTagger"]
