import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around KernalAnsatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

class HybridKernelLSTM(nn.Module):
    """
    Combines a classical RBF kernel with a standard LSTM and a fully‑connected output.
    The kernel maps each input vector to a similarity vector with respect to learnable prototypes.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_prototypes: int = 5, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.kernel = Kernel(kernel_gamma)
        self.gamma = kernel_gamma
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def _kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel features for each element in the sequence.
        x: (batch*seq_len, input_dim)
        Returns: (batch*seq_len, num_prototypes)
        """
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (N, P, D)
        sq_norm = torch.sum(diff * diff, dim=2)  # (N, P)
        return torch.exp(-self.gamma * sq_norm)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        sequence: (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = sequence.shape
        seq_flat = sequence.reshape(-1, sequence.size(-1))  # (batch*seq_len, D)
        features = self._kernel_features(seq_flat)          # (batch*seq_len, P)
        features = features.reshape(batch, seq_len, -1)     # (batch, seq_len, P)
        lstm_out, _ = self.lstm(features)                  # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]                    # (batch, hidden_dim)
        return self.fc(last_hidden)                         # (batch, 1)
