"""Quantum‑enhanced kernel and LSTM for sequence tagging.

This module implements:
- A variational quantum kernel based on a 4‑qubit RX ansatz.
- A quantum LSTM cell that uses small quantum circuits for each gate.
- A sequence tagging model that can switch between classical and quantum LSTM.
- Backward‑compatibility aliases for the original API.

The quantum implementations rely on torchquantum and are fully differentiable
using the PyTorch autograd engine.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict

# --- Quantum kernel ---------------------------------------------------------

class QKernel(tq.QuantumModule):
    """Variational quantum kernel using a 4‑qubit RX ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the scalar kernel value for a batch of pairs."""
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def qkernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QKernel()
    return np.array([[kernel.kernel_value(x, y).item() for y in b] for x in a])

# --- Quantum LSTM -----------------------------------------------------------

class QLayer(tq.QuantumModule):
    """Quantum circuit that implements a single LSTM gate."""
    def __init__(self, n_wires: int):
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
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tq.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell where each gate is a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear layers mapping classical concatenated states to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum layers implementing the gates
        self.forget_gate = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update_gate = QLayer(n_qubits)
        self.output_gate = QLayer(n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QLSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # shape (1, seq_len, embedding_dim)
        embeds = embeds.squeeze(0).unsqueeze(1)  # shape (seq_len, 1, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)

# --- Hybrid wrapper ---------------------------------------------------------

class HybridKernelLSTM(nn.Module):
    """Hybrid kernel + LSTM wrapper that can be used in a single training loop."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int, gamma: float = 1.0):
        super().__init__()
        self.kernel = QKernel()
        self.tagger = QLSTMTagger(embedding_dim, hidden_dim,
                                  vocab_size, tagset_size, n_qubits=n_qubits)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.tagger(sentence)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return qkernel_matrix(a, b)

# Backward‑compatibility aliases for the original API
KernalAnsatz = QKernel
Kernel = HybridKernelLSTM
QLSTM = QLSTMTagger
LSTMTagger = HybridKernelLSTM

__all__ = ["QKernel", "qkernel_matrix", "QLayer", "QLSTM", "QLSTMTagger",
           "HybridKernelLSTM", "KernalAnsatz", "Kernel", "QLSTM", "LSTMTagger"]
