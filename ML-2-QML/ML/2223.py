import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Import the quantum LSTM cell from the seed
from.qlstm import QLSTM

# Import the quantum hybrid head
from.quantum_head import HybridQuantumHead

class ResidualQLSTM(nn.Module):
    """
    Residual LSTM block that can be either classical (nn.LSTM) or quantum (QLSTM).
    The block consists of a stacked LSTM cell with a skip connection that
    normalises the input and output to keep gradients flowing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        if n_qubits > 0:
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters:
            x: (batch, seq_len, feature) or (seq_len, batch, feature)
            states: initial hidden and cell states.
        """
        if isinstance(self.lstm, nn.LSTM):
            out, (h, c) = self.lstm(x, states)
            out = self.norm(out)
            # Residual connection
            return out + x, (h, c)
        else:
            # QLSTM expects a sequence of shape (seq_len, batch, feature)
            seq = x if x.ndim == 3 else x.unsqueeze(0)
            out, (h, c) = self.lstm(seq, states)
            out = self.norm(out)
            return out + seq, (h, c)

class HybridHead(nn.Module):
    """
    Hybrid head that can be classical or quantum.
    """
    def __init__(self, in_features: int, n_qubits: int = 0, shift: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        if n_qubits > 0:
            self.head = HybridQuantumHead(in_features, shift)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class QuantumEnhancedLSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical, quantum, or hybrid modes.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, use_quantum_head: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_block = ResidualQLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.head = HybridHead(tagset_size, n_qubits if use_quantum_head else 0)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm_block(embeds, None)
        tag_logits = self.hidden2tag(lstm_out)
        probs = self.head(tag_logits)
        return F.log_softmax(probs, dim=1)

__all__ = ["QuantumEnhancedLSTMTagger", "ResidualQLSTM", "HybridHead"]
