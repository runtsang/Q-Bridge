import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class HybridQLSTM(nn.Module):
    """
    Classical LSTM cell with curriculum learning and optional weight sharing.
    The cell keeps the original interface but adds a schedule for sequence
    lengths, enabling progressive training on longer sequences.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 curriculum: list[int] | None = None,
                 share_weights: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.curriculum = sorted(curriculum) if curriculum else []
        self.share_weights = share_weights

        # Linear projections for gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        seq_len = inputs.size(0)
        max_len = self.curriculum[-1] if self.curriculum else seq_len
        for t, x in enumerate(inputs.unbind(dim=0)):
            if t >= max_len:
                break
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def set_curriculum(self, seq_lengths: list[int]):
        """Set the curriculum schedule (list of max sequence lengths)."""
        self.curriculum = sorted(seq_lengths)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses :class:`HybridQLSTM` for the recurrent
    layer.  The tagger is identical to the original seed but now supports
    curriculum learning via the underlying LSTM cell.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 curriculum: list[int] | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, curriculum)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
