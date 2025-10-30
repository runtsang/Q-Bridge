import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen313(nn.Module):
    """
    Classical LSTM cell with optional quantum scaling per gate.
    If n_qubits > 0 and a quantum_gate callable is provided,
    the gate activations are multiplied by the quantum scaling factor.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, depth: int = 1,
                 quantum_gate: Optional[callable] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.quantum_gate = quantum_gate
        if n_qubits > 0 and quantum_gate is None:
            # Default scaling factor is 1 (no quantum effect)
            self.quantum_gate = lambda x: torch.ones_like(x)

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))
            if self.n_qubits > 0:
                scale_f = self.quantum_gate(f)
                scale_i = self.quantum_gate(i)
                scale_g = self.quantum_gate(g)
                scale_o = self.quantum_gate(o)
                f = f * scale_f
                i = i * scale_i
                g = g * scale_g
                o = o * scale_o
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTaggerGen313(nn.Module):
    """
    Sequence tagging model that uses QLSTMGen313 or a standard nn.LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, depth: int = 1,
                 quantum_gate: Optional[callable] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen313(embedding_dim, hidden_dim,
                                    n_qubits=n_qubits, depth=depth,
                                    quantum_gate=quantum_gate)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen313", "LSTMTaggerGen313"]
