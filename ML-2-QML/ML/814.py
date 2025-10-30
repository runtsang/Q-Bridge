import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QLSTMGen(nn.Module):
    """Hybrid LSTM with a classical quantum‑inspired layer."""

    class _QLayer(nn.Module):
        """Simple classical approximation of a quantum gate."""
        def __init__(self, hidden_dim: int, depth: int = 1) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.depth = depth
            self.linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Repeatedly apply a linear transform followed by tanh
            for _ in range(self.depth):
                x = torch.tanh(self.linear(x))
            return x

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Classical “quantum” gates
        self.forget = self._QLayer(hidden_dim, depth)
        self.input = self._QLayer(hidden_dim, depth)
        self.update = self._QLayer(hidden_dim, depth)
        self.output = self._QLayer(hidden_dim, depth)

        # Linear projections to the gate space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTaggerGen(nn.Module):
    """Sequence tagging model that can switch between classical and quantum‑inspired LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen", "LSTMTaggerGen"]
