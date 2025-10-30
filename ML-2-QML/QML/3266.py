"""Hybrid Kernel‑LSTM implementation with quantum components.

The module defines a quantum‑enhanced kernel and an LSTM tagger that can
operate in a fully quantum mode.  The quantum kernel is a variational
circuit that can be trained with gradient‑based optimizers.  The
quantum LSTM cell implements each gate as a small quantum circuit.
Both the kernel and the LSTM are exposed through the same
`HybridKernelLSTM` interface as in the classical module.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq

__all__ = [
    "HybridQuantumKernel",
    "HybridKernelLSTM",
    "kernel_matrix",
]

# --------------------------------------------------------------------------- #
# Quantum kernel utilities
# --------------------------------------------------------------------------- #
class HybridQuantumKernel(tq.QuantumModule):
    """Quantum kernel with a trainable variational circuit."""
    def __init__(self, n_wires: int = 4, device: str | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=1, device=device)
        # Simple encoding: one RY per wire
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable parameters for a small variational layer
        self.params = nn.Parameter(torch.randn(n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two 1‑D tensors.

        The kernel is defined as the absolute value of the overlap between
        the quantum states obtained by encoding `x` and `y` with the
        variational circuit.
        """
        # Encode x
        self.q_device.reset_states(1)
        self.encoder(self.q_device, x)
        for i, p in enumerate(self.params):
            tq.RX(has_params=True, trainable=True)(self.q_device, wires=i, params=p)
        state_x = self.q_device.states.clone()
        # Encode y with negative params to obtain overlap
        self.q_device.reset_states(1)
        self.encoder(self.q_device, y)
        for i, p in enumerate(self.params):
            tq.RX(has_params=True, trainable=True)(self.q_device, wires=i, params=-p)
        # Compute overlap amplitude
        overlap = torch.abs(torch.einsum("i,i->", state_x.view(-1), self.q_device.states.view(-1)))
        return overlap

    def gram(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        """Return a Gram matrix for two batches."""
        n = batch_x.shape[0]
        m = batch_y.shape[0]
        out = torch.empty((n, m), device=batch_x.device)
        for i in range(n):
            for j in range(m):
                out[i, j] = self.forward(batch_x[i], batch_y[j])
        return out

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    """Compute a Gram matrix between two iterables of tensors using the quantum kernel."""
    kernel = HybridQuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Quantum LSTM utilities
# --------------------------------------------------------------------------- #
class QQuantumLayer(tq.QuantumModule):
    """Small quantum circuit used as an LSTM gate."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.params = nn.ParameterList([nn.Parameter(torch.randn(n_wires)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, p in enumerate(self.params):
            tq.RX(has_params=True, trainable=True)(qdev, wires=i, params=p)
        # Entangling layer
        for i in range(self.n_wires - 1):
            tq.cnot(qdev, wires=[i, i + 1])
        # Measurement of Pauli‑Z on each wire
        out = []
        for i in range(self.n_wires):
            out.append(tq.Measure(tq.PauliZ, wires=[i])(qdev))
        return torch.cat(out, dim=-1)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = QQuantumLayer(n_qubits)
        self.input = QQuantumLayer(n_qubits)
        self.update = QQuantumLayer(n_qubits)
        self.output = QQuantumLayer(n_qubits)
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Hybrid Kernel + LSTM architecture
# --------------------------------------------------------------------------- #
class HybridKernelLSTM(nn.Module):
    """
    Hybrid architecture that couples a quantum kernel with a quantum LSTM tagger.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Number of tokens in the vocabulary.
    tagset_size : int
        Number of distinct tags.
    n_qubits : int, optional
        Number of qubits used by the kernel and the quantum LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.kernel = HybridQuantumKernel(n_wires=n_qubits)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of token indices with shape (seq_len,).

        Returns
        -------
        torch.Tensor
            Log‑softmax of tag probabilities for each token.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def kernel_gram(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        """Return Gram matrix between two batches using the quantum kernel."""
        return self.kernel.gram(batch_x, batch_y)
