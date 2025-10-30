import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ClassicalGate(nn.Module):
    """Linear gate with standard activation."""
    def __init__(self, in_features: int, out_features: int, act: str = "sigmoid"):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "sigmoid":
            return torch.sigmoid(self.proj(x))
        elif self.act == "tanh":
            return torch.tanh(self.proj(x))
        else:
            raise ValueError(f"Unsupported activation {self.act}")

class QuantumInspiredGate(nn.Module):
    """Simple quantum-inspired gate using sine and cosine transformations."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(in_features, out_features))
        self.phi = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x @ self.theta
        phase = x @ self.phi
        probs = torch.sin(angles) ** 2
        probs = probs + torch.cos(phase) ** 2
        return probs

class HybridGate(nn.Module):
    """Combines classical and quantum-inspired gates with a learnable mixing ratio."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.classical = ClassicalGate(in_features, out_features, act="sigmoid")
        self.quantum = QuantumInspiredGate(in_features, out_features)
        self.mix = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix * self.classical(x) + (1.0 - self.mix) * self.quantum(x)

class HybridQLSTMCell(nn.Module):
    """Hybrid LSTM cell with mixable gates."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_gate = HybridGate(input_dim + hidden_dim, hidden_dim)
        self.input_gate = HybridGate(input_dim + hidden_dim, hidden_dim)
        self.update_gate = HybridGate(input_dim + hidden_dim, hidden_dim)
        self.output_gate = HybridGate(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        f = self.forget_gate(combined)
        i = self.input_gate(combined)
        g = torch.tanh(self.update_gate(combined))
        o = self.output_gate(combined)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

class HybridQLSTM(nn.Module):
    """Wrapper around the HybridQLSTMCell to process sequences."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = HybridQLSTMCell(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hidden is None:
            batch_size = inputs.size(1)
            device = inputs.device
            h = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
            hidden = (h, c)
        outputs = []
        for t in range(inputs.size(0)):
            x_t = inputs[t]
            h, hidden = self.cell(x_t, hidden)
            outputs.append(h.unsqueeze(0))
        return torch.cat(outputs, dim=0), hidden

class LSTMTagger(nn.Module):
    """Sequence tagging model using the hybrid LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=2)
