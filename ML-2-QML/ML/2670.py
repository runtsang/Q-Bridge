"""Hybrid kernel‑LSTM module with a classical backbone and optional quantum enhancements."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalRBFKernel(nn.Module):
    """Pure PyTorch RBF kernel with adjustable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelMatrixBuilder:
    """Builds a Gram matrix from a sequence of tensors using a kernel module."""
    def __init__(self, kernel: nn.Module, device: torch.device | None = None):
        self.kernel = kernel
        self.device = device or torch.device("cpu")

    def __call__(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        self.kernel.to(self.device)
        self.kernel.eval()
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

class QLSTM(nn.Module):
    """Classical LSTM cell that mimics the quantum interface."""
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
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between quantum and classical LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridKernelLSTM(nn.Module):
    """Hybrid model combining a kernel‑based feature map and a quantum‑LSTM layer."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        kernel_type: str = "classical",
        n_qubits: int = 0,
        gamma: float = 1.0,
        n_wires: int = 4,
        n_refs: int = 10,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.n_wires = n_wires
        self.n_refs = n_refs
        if kernel_type == "quantum":
            raise NotImplementedError("Quantum kernel requires torchquantum, use qml_code.")
        else:
            self.kernel = ClassicalRBFKernel(gamma=gamma)
        self.ref_vectors = nn.Parameter(torch.randn(n_refs, embedding_dim))
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def _kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "classical":
            diff = x[:, None, :] - self.ref_vectors[None, :, :]
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))
        else:
            feats = []
            for ref in self.ref_vectors:
                feats.append(self.kernel(x, ref))
            return torch.stack(feats, dim=-1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        feats = self._kernel_features(embeds)
        lstm_input = torch.cat([embeds, feats], dim=-1)
        lstm_out, _ = self.lstm(lstm_input.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = [
    "ClassicalRBFKernel",
    "KernelMatrixBuilder",
    "QLSTM",
    "LSTMTagger",
    "HybridKernelLSTM",
]
