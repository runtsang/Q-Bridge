import numpy as np
import torch
from torch import nn
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel implementation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumKernelMethod(nn.Module):
    """
    Hybrid classical kernel that optionally augments input with a linear LSTM
    before computing a Gaussian RBF. Mirrors the quantum counterpart but remains
    fully classical.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 n_qubits: int = 4,
                 use_lstm: bool = False,
                 lstm_hidden: int = 8) -> None:
        super().__init__()
        self.gamma = gamma
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_size=n_qubits,
                                hidden_size=lstm_hidden,
                                batch_first=True)
            self.lstm_linear = nn.Linear(lstm_hidden, n_qubits)
        else:
            self.lstm = None
        self.kernel = KernalAnsatz(gamma)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Optional LSTM feature transformation."""
        if not self.use_lstm:
            return x
        seq = x.unsqueeze(1)  # (batch, 1, n_qubits)
        out, _ = self.lstm(seq)
        out = out.squeeze(1)
        out = self.lstm_linear(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel between two batches of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Shape (N, D) or (N, D,...) where D is the feature dimension.
            If ``use_lstm`` is True, D must equal ``n_qubits``.
        """
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        if self.use_lstm:
            x = self._transform(x)
            y = self._transform(y)
        return self.kernel(x, y)

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  n_qubits: int = 4,
                  use_lstm: bool = False,
                  lstm_hidden: int = 8) -> np.ndarray:
    """Utility to compute Gram matrix from a list of tensors."""
    kernel = QuantumKernelMethod(gamma=gamma,
                                 n_qubits=n_qubits,
                                 use_lstm=use_lstm,
                                 lstm_hidden=lstm_hidden)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "QuantumKernelMethod", "kernel_matrix"]
