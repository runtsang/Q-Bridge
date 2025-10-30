import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSampler(nn.Module):
    """Approximate a quantum sampler: maps gate logits to a probability vector."""
    def __init__(self, gate_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gate_dim, 4),
            nn.Tanh(),
            nn.Linear(4, gate_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, gate_dim)
        return F.softmax(self.net(x), dim=-1)


class HybridQLSTM(nn.Module):
    """Hybrid LSTM that optionally uses a quantum-inspired sampler for gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.input_dim = input_dim

        # Classical linear projections for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum-inspired sampler to shape gate activations
        if n_qubits > 0:
            self.sampler = ClassicalSampler(hidden_dim)
        else:
            self.sampler = None

    def forward(self, inputs: torch.Tensor,
                states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            # Modulate gates with a quantum-inspired sampler when enabled
            if self.sampler is not None:
                f = self.sampler(f)
                i = self.sampler(i)
                g = self.sampler(g)
                o = self.sampler(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple | None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and hybrid LSTMs."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
