import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Classical RBF kernel used for attention modulation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelMatrix:
    """Utility to compute Gram matrices via RBF kernel."""
    @staticmethod
    def compute(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        k = RBFKernel(gamma)
        return np.array([[k(x, y).item() for y in b] for x in a])

class ClassicalLSTMCell(nn.Module):
    """Standard LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget(combined))
        i = torch.sigmoid(self.input_gate(combined))
        g = torch.tanh(self.update(combined))
        o = torch.sigmoid(self.output(combined))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class HybridQLSTM(nn.Module):
    """Classical hybrid LSTM that modulates the input gate with an RBF kernel attention."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ClassicalLSTMCell(input_dim, hidden_dim)
        self.kernel = RBFKernel(kernel_gamma)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            att = self.kernel(x, hx)
            hx, cx = self.cell(x, hx, cx)
            # Modulate input gate with kernel attention
            hx = hx * att
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class HybridTagger(nn.Module):
    """Sequence tagging model that uses the hybrid LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, kernel_gamma)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridTagger", "KernelMatrix"]
