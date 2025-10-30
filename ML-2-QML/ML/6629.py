import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """
    Classic LSTM cell with optional gate‑activation inspection.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_activations: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[dict]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        activations = [] if return_activations else None
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
            if return_activations:
                activations.append({"f": f, "i": i, "g": g, "o": o})
        stacked = torch.cat(outputs, dim=0)
        if return_activations:
            return stacked, (hx, cx), activations
        return stacked, (hx, cx), None

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses QLSTM.  Supports optional pretrained embeddings.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(pretrained_embeddings)
        self.lstm = QLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(
        self,
        sentence: torch.Tensor,
        return_activations: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _, activations = self.lstm(
            embeds.view(len(sentence), 1, -1),
            return_activations=return_activations
        )
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        log_probs = F.log_softmax(tag_logits, dim=1)
        if return_activations:
            return log_probs, activations
        return log_probs, None

__all__ = ["QLSTM", "LSTMTagger"]
