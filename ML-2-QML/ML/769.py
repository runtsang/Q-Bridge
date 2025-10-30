import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class QLSTMPlus(nn.Module):
    """
    Classical LSTM cell with learnable temperature‑controlled activations and dropout.
    The cell mirrors the quantum interface but remains fully classical.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Linear projections for each gate
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Learnable temperature parameters per gate
        self.temp_forget = Parameter(torch.tensor(1.0))
        self.temp_input = Parameter(torch.tensor(1.0))
        self.temp_update = Parameter(torch.tensor(1.0))
        self.temp_output = Parameter(torch.tensor(1.0))

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_lin(combined) / self.temp_forget)
            i = torch.sigmoid(self.input_lin(combined) / self.temp_input)
            g = torch.tanh(self.update_lin(combined) / self.temp_update)
            o = torch.sigmoid(self.output_lin(combined) / self.temp_output)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the temperature‑controlled QLSTMPlus cell.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMPlus", "LSTMTagger"]
