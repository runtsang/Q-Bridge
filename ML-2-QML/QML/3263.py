"""Quantum‑enhanced kernel and LSTM tagger.

The module contains:
* `VariationalRBFKernel` – a quantum kernel that uses a trainable ansatz to
  produce a data‑dependent feature map.
* `kernel_matrix` – Gram matrix evaluation for quantum kernels.
* `QLSTM` – an LSTM cell where each gate is implemented by a small
  variational quantum circuit.
* `LSTMTagger` – a sequence tagging model that can switch between the
  classical LSTM and the quantum one.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Tuple, Optional

__all__ = [
    "VariationalRBFKernel",
    "kernel_matrix",
    "QLSTM",
    "LSTMTagger",
]


class VariationalRBFKernel(tq.QuantumModule):
    """Quantum RBF kernel based on a variational feature map.

    The circuit encodes the two input vectors `x` and `y` via Ry gates,
    applies a trainable layer of RX rotations, and then measures the
    overlap of the resulting states.  The kernel value is the absolute
    value of the overlap.
    """
    def __init__(self, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Data encoding layers
        self.encoder_x = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.encoder_y = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )

        # Variational layer: trainable RX rotations
        self.var_layer = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        # Entanglement: simple CNOT chain
        self.entangle = nn.ModuleList(
            [tq.CNOT() for _ in range(n_wires - 1)]
        )

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value k(x, y)."""
        self.q_device.reset_states(x.shape[0])

        # Encode x
        self.encoder_x(self.q_device, x)
        # Apply variational layer
        for gate in self.var_layer:
            gate(self.q_device, wires=list(range(self.n_wires)))
        # Entangle
        for idx, gate in enumerate(self.entangle):
            gate(self.q_device, wires=[idx, idx + 1])
        # Measure
        out_x = self.measure(self.q_device).view(-1)

        # Reinitialize device for y
        self.q_device.reset_states(y.shape[0])

        # Encode y
        self.encoder_y(self.q_device, y)
        # Apply same variational layer (shared parameters)
        for gate in self.var_layer:
            gate(self.q_device, wires=list(range(self.n_wires)))
        # Entangle
        for idx, gate in enumerate(self.entangle):
            gate(self.q_device, wires=[idx, idx + 1])
        # Measure
        out_y = self.measure(self.q_device).view(-1)

        # Overlap (inner product) between |x> and |y>
        overlap = torch.abs(torch.dot(out_x, out_y))
        return overlap


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix for two sets of samples using the quantum kernel."""
    kernel = VariationalRBFKernel()
    return np.array(
        [[kernel(x, y).item() for y in b] for x in a]
    )


class QLSTM(tq.QuantumModule):
    """LSTM cell where each gate is a small variational quantum circuit."""
    class QGate(tq.QuantumModule):
        """Encodes a single gate (forget, input, update, output)."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.var = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.entangle = nn.ModuleList(
                [tq.CNOT() for _ in range(n_wires - 1)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.var:
                gate(qdev, wires=list(range(self.n_wires)))
            for idx, gate in enumerate(self.entangle):
                gate(qdev, wires=[idx, idx + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for LSTM
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Classical linear layers to map to qubit dimension
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
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
    """Sequence tagging model that can switch between the classical LSTM
    and the quantum LSTM defined in this module.
    """
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
