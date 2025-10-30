"""Quantum modules for hybrid LSTM tagger.

Provides a quantum convolutional neural network (QCNN) and a quantum LSTM
cell (QLSTM).  Both are implemented using torchquantum and expose a
PyTorch‑compatible interface.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
# Helper: small quantum circuit for convolution or pooling
# --------------------------------------------------------------------------- #
def _conv_circuit(n_wires: int) -> tq.QuantumModule:
    """Return a small variational circuit used as a convolution kernel."""
    class ConvLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # simple entangling pattern
            for i in range(n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    return ConvLayer()


def _pool_circuit(n_wires: int) -> tq.QuantumModule:
    """Return a small variational circuit used as a pooling kernel."""
    class PoolLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # simple measurement after entanglement
            for i in range(n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    return PoolLayer()


# --------------------------------------------------------------------------- #
# Quantum Convolutional Neural Network
# --------------------------------------------------------------------------- #
class QCNN(nn.Module):
    """Quantum convolutional neural network for feature extraction.

    The architecture mirrors the classical QCNN model from the reference
    but replaces each convolution and pooling step with a small
    variational quantum circuit.  The network can be used as a drop‑in
    replacement for a classical feature extractor.
    """

    def __init__(self, input_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Feature map: encode each input feature into a qubit
        self.feature_map = _conv_circuit(n_qubits)

        # Convolution and pooling layers
        self.conv1 = _conv_circuit(n_qubits)
        self.pool1 = _pool_circuit(n_qubits)
        self.conv2 = _conv_circuit(n_qubits)
        self.pool2 = _pool_circuit(n_qubits)
        self.conv3 = _conv_circuit(n_qubits)

        # Final classical head
        self.head = nn.Linear(n_qubits, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, input_dim) where each element is a real number.
        """
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, n_qubits={self.n_qubits})"


# --------------------------------------------------------------------------- #
# Quantum LSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell.

    Each gate of the LSTM is implemented by a small variational quantum
    circuit that operates on ``n_qubits`` qubits.  The classical linear
    layers map the classical input to the qubit space before the quantum
    gates are applied.
    """

    class QGate(tq.QuantumModule):
        """A generic quantum gate block used for LSTM gates."""
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_qubits)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # entangle all qubits
            for i in range(self.n_qubits - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear layers to map concatenated input and hidden state
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates for each LSTM gate
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the quantum LSTM.

        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of inputs with shape (seq_len, batch, input_dim).
        states : tuple, optional
            Initial hidden and cell states.  If ``None`` zero tensors are
            used.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)  # (batch, input+hidden)

            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, n_qubits={self.n_qubits})"


__all__ = ["QCNN", "QLSTM"]
