import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFCLLSTM(nn.Module):
    """
    Classical hybrid model that fuses a fully‑connected layer (FCL) with a
    linear LSTM for sequence tagging.  The FCL maps each token embedding
    into a scalar expectation value which is then treated as the single
    input feature for the LSTM.  The design mirrors the two seed examples
    while providing a clean separation of feature‑mapping and sequence
    modelling.
    """

    class _FCL(nn.Module):
        """Linear mapping that outputs a scalar per token."""
        def __init__(self, input_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (seq_len, batch, input_dim)
            return torch.tanh(self.linear(x))

    def __init__(self, n_features: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, n_features)
        self.fcl = self._FCL(n_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.LongTensor
            Tensor of token indices with shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑softmax scores for each tag at every position.
        """
        embeds = self.word_embeddings(sentence)          # (seq_len, batch, n_features)
        fcl_out = self.fcl(embeds).squeeze(-1)          # (seq_len, batch)
        lstm_out, _ = self.lstm(fcl_out.unsqueeze(-1)) # (seq_len, batch, hidden_dim)
        logits = self.hidden2tag(lstm_out)              # (seq_len, batch, tagset_size)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridFCLLSTM"]
