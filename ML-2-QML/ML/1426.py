import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QLSTM(nn.Module):
    """
    Hybrid quantum‑classical LSTM cell.
    Each gate is computed by a small variational quantum circuit that outputs a scalar activation.
    The circuit is defined by a parameterised RX‑RZ‑CNOT ladder.
    """
    class _QLayer(nn.Module):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Linear map from classical input to qubit amplitudes
            self.lin = nn.Linear(1, n_qubits, bias=False)
            # Parameters for RX and RZ rotations
            self.rz = nn.Parameter(torch.randn(n_qubits))
            # Entanglement parameters (simulate CNOT weights)
            self.cnot = nn.Parameter(torch.randn(n_qubits - 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape (batch, 1)
            z = self.lin(x)  # (batch, n_qubits)
            # apply rotations
            z = torch.cos(z) * torch.cos(self.rz) - torch.sin(z) * torch.sin(self.rz)
            # combine with CNOT‑like entanglement
            for i in range(self.n_qubits - 1):
                z[:, i] += self.cnot[i] * z[:, i + 1]
            return z.mean(-1, keepdim=True)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)

        self.lin_f = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_i = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_g = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_o = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        # inputs shape (batch, seq_len, feature)
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_f(combined)))
            i = torch.sigmoid(self.input(self.lin_i(combined)))
            g = torch.tanh(self.update(self.lin_g(combined)))
            o = torch.sigmoid(self.output(self.lin_o(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        return torch.cat(outputs, dim=1), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses either the hybrid QLSTM or a standard nn.LSTM.
    Includes a gated attention mechanism over the LSTM outputs.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence shape (batch, seq_len)
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        # attention
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        tag_logits = self.hidden2tag(context)
        return F.log_softmax(tag_logits, dim=-1)
