import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMEnhanced(nn.Module):
    """Classical LSTM cell that mirrors the quantum LSTM structure.

    The cell uses the same linear transformations for each gate but replaces
    the quantum circuit with a small feed‑forward network.  This allows
    direct comparison of training dynamics between the classical and quantum
    implementations.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 gate_ff_hidden: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_dim = hidden_dim

        # Linear layers that produce raw gate pre‑activations
        self.linear_forget = nn.Linear(input_dim + hidden_dim, self.gate_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, self.gate_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, self.gate_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, self.gate_dim)

        # Feed‑forward networks that emulate the quantum gates
        self.ff_forget = nn.Sequential(
            nn.Linear(self.gate_dim, gate_ff_hidden),
            nn.ReLU(),
            nn.Linear(gate_ff_hidden, self.gate_dim),
            nn.Sigmoid()
        )
        self.ff_input = nn.Sequential(
            nn.Linear(self.gate_dim, gate_ff_hidden),
            nn.ReLU(),
            nn.Linear(gate_ff_hidden, self.gate_dim),
            nn.Sigmoid()
        )
        self.ff_update = nn.Sequential(
            nn.Linear(self.gate_dim, gate_ff_hidden),
            nn.ReLU(),
            nn.Linear(gate_ff_hidden, self.gate_dim),
            nn.Tanh()
        )
        self.ff_output = nn.Sequential(
            nn.Linear(self.gate_dim, gate_ff_hidden),
            nn.ReLU(),
            nn.Linear(gate_ff_hidden, self.gate_dim),
            nn.Sigmoid()
        )

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.ff_forget(self.linear_forget(combined))
            i = self.ff_input(self.linear_input(combined))
            g = self.ff_update(self.linear_update(combined))
            o = self.ff_output(self.linear_output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, gate_ff_hidden: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(embedding_dim,
                                      hidden_dim,
                                      n_qubits=n_qubits,
                                      gate_ff_hidden=gate_ff_hidden)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMEnhanced", "LSTMTagger"]
