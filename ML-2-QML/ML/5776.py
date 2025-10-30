"""Hybrid kernel and LSTM module for classical experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalRBFKernel(nn.Module):
    """Fast RBF kernel with trainable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernel(nn.Module):
    """Kernel that can use classical RBF or a quantum kernel."""
    def __init__(self, use_quantum: bool = False, gamma: float = 1.0, n_wires: int = 4) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            import torchquantum as tq
            class _QuantumRBF(tq.QuantumModule):
                def __init__(self) -> None:
                    super().__init__()
                    self.encoder = tq.GeneralEncoder(
                        [
                            {"input_idx": [0], "func": "ry", "wires": [0]},
                            {"input_idx": [1], "func": "ry", "wires": [1]},
                            {"input_idx": [2], "func": "ry", "wires": [2]},
                            {"input_idx": [3], "func": "ry", "wires": [3]},
                        ]
                    )

                def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    qdev.reset_states(1)
                    self.encoder(qdev, x)
                    return torch.abs(qdev.states.view(-1)[0])

            self.kernel = _QuantumRBF()
        else:
            self.kernel = ClassicalRBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)


class HybridKernelLSTM(nn.Module):
    """Hybrid model combining a kernel module and an LSTM for sequence tagging."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 gamma: float = 1.0,
                 use_quantum_kernel: bool = False,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.kernel = HybridKernel(use_quantum=use_quantum_kernel, gamma=gamma)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
