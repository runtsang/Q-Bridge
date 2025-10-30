import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class HybridQLSTMKernel(nn.Module):
    """
    Classical implementation of a hybrid LSTM + kernel module.
    The LSTM is a standard nn.LSTM; kernel computations are performed
    with a classical radial‑basis‑function (RBF) kernel.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Use a classical LSTM; the quantum flag is ignored here
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.kernel_gamma = kernel_gamma

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sets of feature vectors using
        a classical radial‑basis‑function kernel.
        """
        return np.array([[self._rbf_kernel(x, y, self.kernel_gamma).item() for y in b] for x in a])

__all__ = ["HybridQLSTMKernel"]
