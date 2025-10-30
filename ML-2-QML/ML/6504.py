import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """Classical softmax sampler mimicking a quantum SamplerQNN."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.Tanh(),
            nn.Linear(4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class QLSTMHybrid(nn.Module):
    """Hybrid LSTM that can operate with classical or quantum‑inspired gates."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # Quantum‑inspired gating: small MLPs per gate that output gate probabilities.
            self.forget_gate = nn.Sequential(
                nn.Linear(embedding_dim + hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.input_gate = nn.Sequential(
                nn.Linear(embedding_dim + hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.update_gate = nn.Sequential(
                nn.Linear(embedding_dim + hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.output_gate = nn.Sequential(
                nn.Linear(embedding_dim + hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        else:
            # Pure classical LSTM
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.sampler = SamplerModule(tagset_size, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if hasattr(self, 'lstm'):
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        else:
            hx = torch.zeros(embeds.size(1), self.hidden_dim, device=embeds.device)
            cx = torch.zeros(embeds.size(1), self.hidden_dim, device=embeds.device)
            outputs = []
            for x in embeds.unbind(dim=0):
                combined = torch.cat([x, hx], dim=1)
                f = self.forget_gate(combined)
                i = self.input_gate(combined)
                g = self.update_gate(combined)
                o = self.output_gate(combined)
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            lstm_out = torch.cat(outputs, dim=0)
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        probs = self.sampler(tag_logits)
        return torch.log(probs + 1e-12)

__all__ = ["QLSTMHybrid"]
