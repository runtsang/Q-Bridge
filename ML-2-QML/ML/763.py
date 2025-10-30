import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen024(nn.Module):
    """
    Classical LSTM cell inspired by a quantum architecture.
    Each gate is computed by a small neural network that mimics a variational
    quantum circuit. The design keeps the original interface while allowing
    future quantum substitution via the `use_quantum` flag.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, use_quantum: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical linear layers for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional classical variational layer that mimics quantum output
        if self.use_quantum:
            # Shared small MLP to emulate quantum circuit output
            self.var_network = nn.Sequential(
                nn.Linear(n_qubits, n_qubits),
                nn.ReLU(),
                nn.Linear(n_qubits, n_qubits)
            )

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1) if inputs.dim() == 3 else inputs.size(0)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim) or (batch, seq_len, input_dim)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        # iterate over sequence dimension
        if inputs.dim() == 3:
            seq_len = inputs.size(0)
            for t in range(seq_len):
                x = inputs[t]
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

                if self.use_quantum and self.n_qubits > 0:
                    # emulate quantum gate output
                    q_out = self.var_network(torch.randn(self.n_qubits, device=inputs.device))
                    f = f + q_out[:self.hidden_dim]
                    i = i + q_out[:self.hidden_dim]
                    g = g + q_out[:self.hidden_dim]
                    o = o + q_out[:self.hidden_dim]

                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
        else:
            # fallback: treat as (batch, seq_len, input_dim)
            seq_len = inputs.size(1)
            for t in range(seq_len):
                x = inputs[:, t, :]
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

                if self.use_quantum and self.n_qubits > 0:
                    q_out = self.var_network(torch.randn(self.n_qubits, device=inputs.device))
                    f = f + q_out[:self.hidden_dim]
                    i = i + q_out[:self.hidden_dim]
                    g = g + q_out[:self.hidden_dim]
                    o = o + q_out[:self.hidden_dim]

                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(1))
            outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)

class LSTMTaggerGen024(nn.Module):
    """
    Sequence tagging model that can optionally use the hybrid QLSTMGen024.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, use_quantum: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum and n_qubits > 0:
            self.lstm = QLSTMGen024(embedding_dim, hidden_dim, n_qubits=n_qubits, use_quantum=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (batch, seq_len)
        """
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds.transpose(0,1))
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMGen024", "LSTMTaggerGen024"]
