"""Hybrid kernel and LSTM module with quantum components."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridKernelLSTM"]


class QuantumKernel(nn.Module):
    """Variational quantum kernel built on TorchQuantum."""

    def __init__(self, n_wires: int = 4, gate_list: list | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        if gate_list is None:
            gate_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.GeneralEncoder(gate_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute amplitude of the |0⟩ state after encoding x and -y."""
        # Ensure both inputs are of shape (batch, dim)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x)
        self.ansatz(self.q_device, -y)
        # Return the magnitude of the amplitude of |0⟩
        return torch.abs(self.q_device.states[0]).real


class KernelWrapper(nn.Module):
    """Build a Gram matrix from a quantum kernel module."""
    def __init__(self, kernel: nn.Module) -> None:
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Return Gram matrix as numpy array."""
        gram = torch.zeros(len(x), len(y), dtype=torch.float32, device=x.device)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                gram[i, j] = self.kernel(xi, yj)
        return gram.cpu().numpy()


class QLSTM(nn.Module):
    """LSTM cell where gates are realized by small quantum circuits."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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
        states: tuple | None,
    ) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridKernelLSTM(nn.Module):
    """Hybrid module that offers a quantum kernel and a quantum LSTM tagger."""
    def __init__(
        self,
        *,
        n_wires: int = 4,
        embedding_dim: int = 50,
        hidden_dim: int = 128,
        vocab_size: int = 10000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.kernel = KernelWrapper(QuantumKernel(n_wires=n_wires))
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_wires)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Return the quantum kernel Gram matrix."""
        return self.kernel(a, b)

    def tag_sequence(self, sentence: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging using a quantum LSTM.

        Parameters
        ----------
        sentence : LongTensor, shape (seq_len,)
            Token indices of the input sentence.

        Returns
        -------
        log_probs : Tensor, shape (seq_len, tagset_size)
            Log‑probabilities over tags for each token.
        """
        embeds = self.embedding(sentence).unsqueeze(0)  # batch size 1
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out.squeeze(0))
        return F.log_softmax(logits, dim=1)
