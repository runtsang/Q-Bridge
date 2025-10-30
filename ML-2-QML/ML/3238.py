import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSampler(nn.Module):
    """Feed‑forward sampler that maps a hidden vector to a log‑probability distribution."""
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, vocab_size)
        )
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(hidden), dim=-1)

class HybridSamplerQLSTM(nn.Module):
    """
    Classical hybrid sampler.
    Uses a standard LSTM encoder and a classical sampler head.
    The `n_qubits` argument is accepted for API compatibility but
    is ignored in this purely classical implementation.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.sampler = ClassicalSampler(hidden_dim, vocab_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len,) containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑probabilities over the vocabulary for each position.
        """
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # (1, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embeds)  # (1, seq_len, hidden_dim)
        logits = self.sampler(lstm_out.squeeze(0))  # (seq_len, vocab_size)
        return logits
