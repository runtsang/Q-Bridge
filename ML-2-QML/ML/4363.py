import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


class ClassicalSampler(nn.Module):
    """Simple 2‑dimensional softmax sampler."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class ClassicalQLSTM(nn.Module):
    """Drop‑in classical LSTM with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        seq: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(seq, states)
        outputs = []
        for x in seq.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        seq: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = seq.size(1)
        device = seq.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )


class FraudClassifier(nn.Module):
    """Two‑layer Tanh network ending in a single logit."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HybridSamplerQLSTM(nn.Module):
    """Classical hybrid: sampler → LSTM → fraud classifier."""
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.sampler = ClassicalSampler()
        self.lstm = ClassicalQLSTM(input_dim=2, hidden_dim=hidden_dim)
        self.classifier = FraudClassifier()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: Tensor of shape (seq_len, batch, 2)
        """
        probs = self.sampler(inputs)
        lstm_out, _ = self.lstm(probs)
        logits = self.classifier(lstm_out)
        return logits


__all__ = ["HybridSamplerQLSTM"]
