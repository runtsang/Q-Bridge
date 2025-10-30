import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QuantumEnhancedQLSTM(nn.Module):
    """Classical LSTM cell with optional adaptive gates and state noise.

    The module is a drop‑in replacement for the original QLSTM.  It adds:
    * Gaussian noise to the initial hidden and cell states.
    * Trainable weight matrices for each gate that can be optionally
      shared across gates.
    * A configurable dropout on the hidden state.
    * A helper method to compute an L1 sparsity regulariser.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        shared_gate_weights: bool = True,
        noise_std: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.shared_gate_weights = shared_gate_weights
        self.noise_std = noise_std
        self.dropout = dropout

        if shared_gate_weights:
            self.gate = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.drop = nn.Dropout(p=dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            combined = self.drop(combined)
            if self.shared_gate_weights:
                gates = self.gate(combined)
                f, i, g, o = gates.chunk(4, dim=1)
            else:
                f = self.forget(combined)
                i = self.input(combined)
                g = self.update(combined)
                o = self.output(combined)

            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

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
        if self.noise_std > 0:
            noise = torch.randn_like(hx) * self.noise_std
            hx = hx + noise
            cx = cx + noise
        return hx, cx

    def regularisation_loss(self) -> torch.Tensor:
        """Return an L1 sparsity penalty over all trainable parameters."""
        l1 = torch.tensor(0.0, device=self.parameters().__next__().device)
        for p in self.parameters():
            l1 = l1 + torch.norm(p, p=1)
        return l1

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        shared_gate_weights: bool = True,
        noise_std: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumEnhancedQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                shared_gate_weights=shared_gate_weights,
                noise_std=noise_std,
                dropout=dropout,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QuantumEnhancedQLSTM", "LSTMTagger"]
