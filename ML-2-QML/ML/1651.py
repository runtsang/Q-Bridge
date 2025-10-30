import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

class QLSTMTaggerEnhanced(nn.Module):
    """
    Classical LSTM tagger with dropout, residual connections and optional quantum regularisation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        dropout: float = 0.1,
        residual: bool = True,
        reg_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.reg_func = reg_func

        if residual and embedding_dim!= hidden_dim:
            self.res_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.res_proj = None

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Input sentence as a LongTensor of shape (batch, seq_len).

        Returns
        -------
        log_probs : torch.Tensor
            Log-probabilities for each token.
        reg_term : Optional[torch.Tensor]
            Quantum regularisation term if a reg_func is supplied.
        """
        embeds = self.word_embeddings(sentence)
        out, _ = self.lstm(embeds)
        out = self.dropout(out)

        if self.residual:
            res = self.res_proj(embeds) if self.res_proj is not None else embeds
            out = out + res

        logits = self.hidden2tag(out)
        log_probs = F.log_softmax(logits, dim=-1)

        reg_term = self.reg_func(out) if self.reg_func is not None else None
        return log_probs, reg_term

    def compute_regulatory_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that applies the optional quantum regulariser.
        """
        if self.reg_func is None:
            raise ValueError("No regulariser function provided.")
        return self.reg_func(hidden_states)

__all__ = ["QLSTMTaggerEnhanced"]
