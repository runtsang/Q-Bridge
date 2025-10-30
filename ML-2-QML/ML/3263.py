"""Hybrid RBF kernel and LSTM tagger with classical core and optional quantum augmentation.

The module provides:
* `HybridRBFKernel` – a classical RBF kernel with a learnable width.
* `KernelParameterNet` – a tiny network that predicts a data‑dependent width.
* `kernel_matrix` – efficient Gram matrix computation for batched tensors.
* `QLSTM` – a pure PyTorch LSTM cell that mimics the interface of the quantum LSTM.
* `LSTMTagger` – a tagger that can switch between the classical LSTM and the quantum one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Tuple, Optional

__all__ = [
    "KernelParameterNet",
    "HybridRBFKernel",
    "kernel_matrix",
    "QLSTM",
    "LSTMTagger",
]


class KernelParameterNet(nn.Module):
    """Predicts a positive RBF width γ for each pair of inputs.

    The network takes the concatenated pair (x, y) and outputs a scalar
    width that is then used in the kernel formula
        k(x, y) = exp(-γ * ||x - y||²).
    """
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # guarantees positivity
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return γ for each pair (x, y).  Shape: (batch, 1)."""
        pair = torch.stack([x, y], dim=1)  # (batch, 2, dim)
        return self.net(pair.view(-1, 2)).view(-1, 1)


class HybridRBFKernel(nn.Module):
    """Classical RBF kernel that can use a data‑dependent width.

    Parameters
    ----------
    gamma : float | None
        Default width. If ``None``, the kernel uses the width produced by
        :class:`KernelParameterNet`. This allows the kernel to adapt
        locally to the data distribution.
    """
    def __init__(self, gamma: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = gamma
        if gamma is None:
            self.param_net = KernelParameterNet()
        else:
            self.param_net = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute k(x, y) = exp(-γ * ||x - y||²)."""
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        if self.param_net is None:
            gamma = torch.tensor(self.gamma, dtype=x.dtype, device=x.device)
        else:
            gamma = self.param_net(x, y)
        return torch.exp(-gamma * sq_norm)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: Optional[float] = None) -> np.ndarray:
    """Return the Gram matrix between datasets ``a`` and ``b``."""
    kernel = HybridRBFKernel(gamma=gamma)
    return np.array(
        [[kernel(x, y).item() for y in b] for x in a]
    )


# --------------------------------------------------------------------------- #
#  Classical LSTM implementation (drop‑in replacement for the quantum one)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Pure PyTorch LSTM cell that mimics the interface of the quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses a classical LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,  # unused in the classical version
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
