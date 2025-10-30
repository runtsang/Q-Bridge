import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(nn.Module):
    """
    Classical LSTM cell that optionally delegates gate computation to a quantum‑inspired
    non‑linear transformation. When ``use_quantum`` is False the cell behaves as a
    standard LSTM; when True a small feed‑forward network is applied to each gate
    to mimic the expressivity of a quantum circuit while staying fully classical.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, use_quantum: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Linear projections for all gates (input + hidden)
        self.forget_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_proj  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_quantum:
            # Quantum‑inspired block: a small feed‑forward network that
            # emulates parameterized rotations and entanglement.
            self.quantum_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.quantum_block = nn.Identity()

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional tuple (h, c) each of shape (batch, hidden_dim)

        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            final states: (h, c)
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, h], dim=1)

            f = torch.sigmoid(self.forget_proj(combined))
            i = torch.sigmoid(self.input_proj(combined))
            g = torch.tanh(self.update_proj(combined))
            o = torch.sigmoid(self.output_proj(combined))

            # Apply quantum‑inspired block to each gate
            f = self.quantum_block(f)
            i = self.quantum_block(i)
            g = self.quantum_block(g)
            o = self.quantum_block(o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTaggerGen(nn.Module):
    """
    Sequence tagging model that uses either :class:`QLSTMGen` or a vanilla
    ``nn.LSTM`` depending on the ``use_quantum`` flag.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, use_quantum: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum and n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, use_quantum=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
