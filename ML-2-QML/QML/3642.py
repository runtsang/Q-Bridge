"""Hybrid quantum‑classical kernel and LSTM implementation.

This module mirrors the classical version but replaces the kernel with a
parameter‑free quantum circuit and the LSTM gates with small quantum
sub‑circuits. The main class `HybridKernelLSTM` exposes the same API as the
classical counterpart, with a `mode` flag that selects between purely
classical and quantum components.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Tuple

__all__ = [
    "HybridKernelLSTM",
    "Kernel",
    "kernel_matrix",
    "QLSTM",
    "LSTMTagger",
    "KernalAnsatz",
    "ClassicalKernel",
]

# ----------------------------- Classical kernel utilities -----------------------------
class ClassicalKernel(nn.Module):
    """Simple RBF kernel used when operating in classical mode."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# ----------------------------- Quantum kernel utilities -----------------------------
class KernalAnsatz(tq.QuantumModule):
    """Quantum‑encoded RBF‑style kernel using a fixed circuit."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder: encode each input dimension as an Ry rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode data and compute overlap."""
        q_device.reset_states(x.shape[0])
        self.encoder(q_device, x)
        # Encode inverse of y (negative angles)
        inv_encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        inv_encoder(q_device, -y)

class Kernel(tq.QuantumModule):
    """Quantum kernel module that returns a scalar from the overlap of two states."""
    def __init__(self, n_wires: int = 4, device: str = "cpu") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.qdevice = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=device)
        self.ansatz = KernalAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.qdevice, x, y)
        # Return absolute value of overlap (amplitude of |0…0> state)
        return torch.abs(self.qdevice.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute quantum Gram matrix for two sets of inputs."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------- Quantum LSTM implementation -----------------------------
class QLSTM(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encode input as rotations
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            # Trainable rotation gates
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            # Measurement of all qubits in Z basis
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle all qubits via a CNOT chain
            for wire in range(self.n_wires):
                target = wire + 1 if wire + 1 < self.n_wires else 0
                tqf.cnot(qdev, wires=[wire, target])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Each gate is a QLayer
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections to gate dimension
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical LSTM or a quantum LSTM."""
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

# ----------------------------- Unified interface -----------------------------
class HybridKernelLSTM(nn.Module):
    """
    Unified interface combining kernel evaluation and sequence tagging.

    Parameters
    ----------
    mode : str, optional
        'classical' or 'quantum'. Default is 'quantum'.
    n_wires : int, optional
        Number of qubits for the quantum kernel and quantum gates.
    kernel_gamma : float, optional
        RBF kernel width for the classical kernel (ignored in quantum mode).
    embedding_dim, hidden_dim, vocab_size, tagset_size : int, optional
        Hyper‑parameters for the embedding and LSTM components.
    """
    def __init__(
        self,
        mode: str = "quantum",
        n_wires: int = 4,
        kernel_gamma: float = 1.0,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        vocab_size: int = 5000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "classical":
            # Use classical kernel and classical LSTM
            self.kernel = ClassicalKernel(kernel_gamma)
            self.tagger = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0)
        else:
            # Quantum mode
            self.kernel = Kernel(n_wires)
            self.tagger = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=n_wires)

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return scalar kernel value for a pair of samples."""
        return self.kernel(x, y)

    def compute_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    def tag_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        """Return log‑probabilities over tags for each token."""
        return self.tagger(sentence)
