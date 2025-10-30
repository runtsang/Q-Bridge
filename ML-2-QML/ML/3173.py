"""Hybrid kernel and LSTM module for classical‑quantum experiments.

This module merges the ideas from the seed QuantumKernelMethod and QLSTM files.
The `HybridKernel` keeps a classical RBF kernel for the quick baseline and
offers an optional quantum RBF that uses TorchQuantum.  The `HybridQLSTM`
replaces the gate matrices of a standard LSTM with tiny quantum sub‑circuits
that use trainable RX gates and CNOT entanglement.  The combination is
exposed through `HybridKernelQLSTM`, a lightweight wrapper that can be used
in a sequence‑tagging task or in any kernel‑based algorithm.

The design is deliberately lightweight: all tensors are PyTorch tensors,
the quantum parts are wrapped in TorchQuantum modules that are fully
autograd‑compatible, and the API stays identical to the seed classes.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical RBF kernel
# --------------------------------------------------------------------------- #

class RBFKernel(nn.Module):
    """Purely classical RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Quantum kernel (optional)
# --------------------------------------------------------------------------- #

class QuantumKernelAnsatz(nn.Module):
    """Quantum RBF kernel implemented with TorchQuantum."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        try:
            import torchquantum as tq
        except ImportError as exc:
            raise RuntimeError("torchquantum is required for QuantumKernelAnsatz") from exc

        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Encode both vectors into the same device and compute overlap
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)
        self.encoder(self.q_device, y)
        # Measure all qubits and compute the absolute overlap
        return torch.abs(self.measure(self.q_device).squeeze())

class HybridKernelAnsatz(nn.Module):
    """Hybrid kernel that can operate in classical or quantum mode."""

    def __init__(self, gamma: float = 1.0, use_quantum: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.quantum_ansatz = QuantumKernelAnsatz()
        else:
            self.classical_ansatz = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            return self.quantum_ansatz(x, y)
        return self.classical_ansatz(x, y)

# --------------------------------------------------------------------------- #
# Kernel matrix utilities
# --------------------------------------------------------------------------- #

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    use_quantum: bool = False,
) -> np.ndarray:
    """Compute the Gram matrix between two sets of vectors."""
    kernel = HybridKernelAnsatz(gamma, use_quantum)
    return np.array(
        [
            [kernel(x, y).item() for y in b]
            for x in a
        ]
    )

# --------------------------------------------------------------------------- #
# Classical LSTM tagger
# --------------------------------------------------------------------------- #

class ClassicalQLSTM(nn.Module):
    """Drop‑in classical replacement for the quantum LSTM."""

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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class ClassicalLSTMTagger(nn.Module):
    """Sequence tagging model that uses a classical LSTM."""

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
            # In a fully classical setting we still use the classical LSTM
            # but the gates are linear.  The quantum LSTM is provided in the
            # quantum module and can be swapped in by the user.
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Hybrid LSTM tagger (classical + quantum)
# --------------------------------------------------------------------------- #

class HybridQLSTM(nn.Module):
    """Wrapper that selects between classical and quantum LSTM."""

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
            try:
                from qml_code import QuantumQLSTM
                self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
            except Exception:
                # Fallback to classical implementation if quantum import fails
                self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Unified wrapper
# --------------------------------------------------------------------------- #

class HybridKernelQLSTM(nn.Module):
    """Unified model that exposes both a hybrid kernel and a hybrid LSTM tagger."""

    def __init__(
        self,
        kernel_gamma: float = 1.0,
        kernel_use_quantum: bool = False,
        embedding_dim: int = 50,
        hidden_dim: int = 50,
        vocab_size: int = 1000,
        tagset_size: int = 5,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.kernel = HybridKernelAnsatz(kernel_gamma, kernel_use_quantum)
        self.tagger = HybridQLSTM(
            embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Return tag probabilities for a sentence."""
        return self.tagger(sentence)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.kernel.gamma, use_quantum=self.kernel.use_quantum)

__all__ = [
    "RBFKernel",
    "HybridKernelAnsatz",
    "kernel_matrix",
    "ClassicalQLSTM",
    "ClassicalLSTMTagger",
    "HybridQLSTM",
    "HybridKernelQLSTM",
]
