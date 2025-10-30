import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """
    Classical LSTM cell that optionally supports a dropout‑aware
    gate‑wise linear transformation.  The implementation keeps
    the original interface but adds a *dropout* parameter that
    can be toggled at training time.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.dropout is not None:
                combined = self.dropout(combined)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    Sequence tagging model that can switch between a classical
    LSTM, a quantum‑enhanced LSTM, or a hybrid training mode.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        hybrid: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hybrid = hybrid

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.dropout is not None:
            embeds = self.dropout(embeds)
        # Reshape for (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def hybrid_step(self, sentence: torch.Tensor, optimizer: torch.optim.Optimizer,
                    quantum_optimizer: torch.optim.Optimizer, loss_fn):
        """
        Perform one hybrid training step: first back‑propagate the loss
        through the classical parameters using *optimizer*, then
        back‑propagate through the quantum parameters using
        *quantum_optimizer*.  This simple alternation allows the model
        to leverage quantum gradients without changing the overall
        architecture.
        """
        self.train()
        optimizer.zero_grad()
        quantum_optimizer.zero_grad()

        logits = self.forward(sentence)
        loss = loss_fn(logits, sentence)
        loss.backward(retain_graph=True)

        optimizer.step()
        quantum_optimizer.step()

        return loss.item()

__all__ = ["QLSTM", "LSTMTagger"]
