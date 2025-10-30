import torch
import torch.nn as nn
import torch.nn.functional as F

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz used as a gate modifier."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes the ansatz as a callable kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

class HybridQLSTM(nn.Module):
    """LSTM cell with gates augmented by a classical kernel."""
    def __init__(self, input_dim: int, hidden_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel = Kernel(gamma)

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _gate(self, linear: nn.Linear, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, hx], dim=1)
        lin_out = linear(combined)
        k_val = self.kernel(x, hx).unsqueeze(1).repeat(1, self.hidden_dim)
        return torch.sigmoid(lin_out + k_val)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            f = self._gate(self.forget_linear, x, hx)
            i = self._gate(self.input_linear, x, hx)
            g = torch.tanh(self.update_linear(torch.cat([x, hx], dim=1)))
            o = self._gate(self.output_linear, x, hx)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that uses the HybridQLSTM cell."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, gamma: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, gamma=gamma)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
