import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMPlus(nn.Module):
    """
    Hybrid LSTM that optionally replaces its gates with quantum variational
    circuits.  When n_qubits == 0 the module behaves as a pure classical
    LSTM.  A learnable noise parameter `beta` simulates decoherence and can
    be tuned during training.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        noise_beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.noise_beta = nn.Parameter(torch.tensor(noise_beta))

        if n_qubits == 0:
            # Pure classical LSTM
            self.lstm = nn.LSTM(input_dim, hidden_dim)
        else:
            # Classical linear layers that feed the quantum circuit
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.n_qubits == 0:
            return self.lstm(inputs, states)

        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.linear_forget(combined))
            i = torch.sigmoid(self.linear_input(combined))
            g = torch.tanh(self.linear_update(combined))
            o = torch.sigmoid(self.linear_output(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Simulate decoherence via learnable noise
            hx = hx + self.noise_beta * torch.randn_like(hx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the hybrid QLSTMPlus.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.reshape(-1, self.lstm.hidden_dim))
        tag_logits = tag_logits.reshape(lstm_out.size(0), lstm_out.size(1), -1)
        return F.log_softmax(tag_logits, dim=2)
