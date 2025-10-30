import torch
from torch import nn
from torch.nn import functional as F

class LSTMTagger(nn.Module):
    """
    Classical sequence tagger using an LSTM.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridEstimator(nn.Module):
    """
    Combines a lightweight MLP regressor and a sequence tagger.
    """
    def __init__(self,
                 reg_input_dim: int = 2,
                 reg_hidden: tuple = (8, 4),
                 lstm_embed_dim: int = 50,
                 lstm_hidden_dim: int = 100,
                 vocab_size: int = 5000,
                 tagset_size: int = 10):
        super().__init__()
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(reg_input_dim, reg_hidden[0]),
            nn.Tanh(),
            nn.Linear(reg_hidden[0], reg_hidden[1]),
            nn.Tanh(),
            nn.Linear(reg_hidden[1], 1),
        )
        # Tagging head
        self.tagger = LSTMTagger(lstm_embed_dim, lstm_hidden_dim, vocab_size, tagset_size)

    def forward_regression(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

    def forward_tagging(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.tagger(sentence)

    def forward(self, x: torch.Tensor, mode: str = "regression") -> torch.Tensor:
        if mode == "regression":
            return self.forward_regression(x)
        elif mode == "tagging":
            return self.forward_tagging(x)
        else:
            raise ValueError(f"Unsupported mode {mode}")

__all__ = ["HybridEstimator", "LSTMTagger"]
