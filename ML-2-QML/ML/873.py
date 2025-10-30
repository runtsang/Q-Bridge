import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
# Import the quantum hidden‑state updater from the QML module
from.quantum_hidden import QuantumHiddenState

class QLSTMGen(nn.Module):
    """
    Hybrid LSTM where classical linear gates control the flow of information,
    while the hidden state is updated by a variational quantum circuit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum hidden‑state updater
        self.quantum_hidden = QuantumHiddenState(hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional tuple (h, c) of hidden and cell states.
        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            final_states: Tuple (h, c)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):  # iterate over time steps
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))

            cx = f * cx + i * g
            # Quantum update of hidden state
            hx = self.quantum_hidden(hx, cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTagger"]
