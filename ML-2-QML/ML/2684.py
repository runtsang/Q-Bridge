import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention:
    """Classical self‑attention that mimics the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridQLSTM(nn.Module):
    """Hybrid LSTM that can optionally apply classical self‑attention before the recurrent layer."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_attention: bool = False,
                 rotation_params: np.ndarray | None = None,
                 entangle_params: np.ndarray | None = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.use_attention = use_attention
        if use_attention:
            if rotation_params is None or entangle_params is None:
                raise ValueError("Attention parameters must be supplied when use_attention=True")
            self.attention = ClassicalSelfAttention(embedding_dim)
            self.rotation_params = rotation_params
            self.entangle_params = entangle_params

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.use_attention:
            # Convert embeddings to numpy for the attention routine
            embeds_np = embeds.detach().cpu().numpy()
            attn_out = self.attention.run(self.rotation_params, self.entangle_params, embeds_np)
            embeds = torch.tensor(attn_out, device=embeds.device, dtype=embeds.dtype)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridQLSTM"]
