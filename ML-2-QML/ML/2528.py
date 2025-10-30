import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelAnsatz(nn.Module):
    """Trainable RBF kernel with a learnable center and bandwidth."""
    def __init__(self, gamma: float = 1.0, center: torch.Tensor | None = None):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        if center is None:
            self.center = nn.Parameter(torch.randn(1, 1))
        else:
            self.center = nn.Parameter(center)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that exposes a scalar kernel value."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()


class HybridQLSTM(nn.Module):
    """Classical LSTM with RBF‑kernel‑modulated gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_centers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear pre‑gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Kernel module and learnable center
        self.kernel = Kernel()
        self.center = nn.Parameter(torch.randn(1, hidden_dim))

    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Kernel similarity between `x` and the learnable center."""
        return self.kernel(x, self.center)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            k = self._kernel(combined)
            f = torch.sigmoid(self.forget_linear(combined) * k)
            i = torch.sigmoid(self.input_linear(combined) * k)
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined) * k)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM`."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_centers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_centers=n_centers)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
