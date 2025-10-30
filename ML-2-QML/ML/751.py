import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional gradient clipping and early‑stopping support.
    The public API mirrors the original QLSTM class but adds a `train_step` helper.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # The `n_qubits` argument is retained for API compatibility; it is ignored
        # in the classical implementation but can be used to toggle a quantum mode
        # in future extensions.
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

    def train_step(self, inputs, targets, criterion, optimizer,
                   clip_norm: float = 0.0, early_stop_patience: int = None,
                   early_stop_counter: int = None):
        """
        Performs a single training step and returns the loss.
        If `clip_norm` > 0, gradients are clipped.
        If `early_stop_patience` is set, the method updates `early_stop_counter`
        and returns a flag indicating whether to stop training.
        """
        optimizer.zero_grad()
        outputs, _ = self(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if clip_norm > 0.0:
            clip_grad_norm_(self.parameters(), clip_norm)
        optimizer.step()
        stop = False
        if early_stop_patience is not None:
            if hasattr(self, "_early_stop_loss") and loss.item() > self._early_stop_loss:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    stop = True
            else:
                early_stop_counter = 0
                self._early_stop_loss = loss.item()
        return loss.item(), stop, early_stop_counter

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses either the classical QLSTM or nn.LSTM."""

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
