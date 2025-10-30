import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ClassicalQLSTM(nn.Module):
    """Pure PyTorch LSTM cell that mimics the interface of the quantum version."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

class ClassicalHead(nn.Module):
    """Simple dense head with sigmoid activation."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = x.view(x.size(0), -1)
        return torch.sigmoid(logits + self.shift)

class UnifiedQLSTMClassifier(nn.Module):
    """Sequence tagger that can switch between classical and quantum LSTM,
    and ends with a hybrid head."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, head_shift: float = 0.0) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # In this classical module we ignore quantum gates and use classical LSTM
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.head = ClassicalHead(tagset_size, shift=head_shift)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        probs = self.head(tag_logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["UnifiedQLSTMClassifier", "ClassicalHead", "ClassicalQLSTM"]
