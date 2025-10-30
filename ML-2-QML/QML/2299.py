"""Hybrid quantum‑classical architecture that fuses a quantum‑enhanced quanvolution filter with a quantum‑augmented LSTM.

The module defines:
- QuanvolutionFilter: a quantum kernel that maps 2‑x2 image patches to a 4‑dimensional feature vector.
- QuanvolutionClassifier: a simple classifier that uses the quantum filter followed by a linear head.
- QuanvolutionQLSTM: a drop‑in replacement that accepts a sequence of images, applies the quantum filter to each image, and feeds the resulting feature vectors into a quantum LSTM before projecting to tag logits.

Typical usage::

    >>> from quanvolution_gen101 import QuanvolutionQLSTM
    >>> model = QuanvolutionQLSTM(input_dim=1, hidden_dim=64, vocab_size=10000, tagset_size=10, conv_quantum=True, lstm_quantum=True)
    >>> logits = model(x)  # x shape: (batch, seq_len, 1, 28, 28)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2x2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# --------------------------------------------------------------------------- #
# Classical fallback filter (kept for API compatibility)
# --------------------------------------------------------------------------- #
class ClassicalQuanvolutionFilter(nn.Module):
    """Fallback classical filter for use when quantum backend is unavailable."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Quantum‑classical classifier
# --------------------------------------------------------------------------- #
class QuanvolutionClassifier(nn.Module):
    """Simple classifier that uses the quanvolution filter followed by a linear head."""
    def __init__(self, conv_quantum: bool = False) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter() if conv_quantum else ClassicalQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
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

# --------------------------------------------------------------------------- #
# Classical fallback LSTM (kept for API compatibility)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Fallback classical LSTM for use when quantum backend is unavailable."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, inputs: torch.Tensor, states=None):
        return self.lstm(inputs, states)

# --------------------------------------------------------------------------- #
# Combined quantum‑classical architecture
# --------------------------------------------------------------------------- #
class QuanvolutionQLSTM(nn.Module):
    """
    Hybrid sequence classifier that combines a quantum quanvolution filter with a quantum LSTM.
    The input is expected to be a batch of sequences of images with shape
    (batch, seq_len, 1, 28, 28).  The filter is applied to each image in the
    sequence, producing a feature vector that is fed into a quantum LSTM.  Finally a
    linear layer projects the LSTM outputs to tag logits.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector (default 4*14*14).
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary (unused in the quantum version but kept for API compatibility).
    tagset_size : int
        Number of output tags.
    conv_quantum : bool, optional
        Flag to enable the quantum quanvolution filter.  If False, a classical fallback is used.
    lstm_quantum : bool, optional
        Flag to enable the quantum LSTM.  If False, a classical fallback is used.
    """
    def __init__(
        self,
        input_dim: int = 4 * 14 * 14,
        hidden_dim: int = 64,
        vocab_size: int = 10000,
        tagset_size: int = 10,
        conv_quantum: bool = True,
        lstm_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter() if conv_quantum else ClassicalQuanvolutionFilter()
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=hidden_dim) if lstm_quantum else ClassicalQLSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (seq_len, batch, tagset_size).
        """
        batch, seq_len, c, h, w = x.shape
        # Flatten the sequence dimension into the batch dimension
        x_flat = x.view(batch * seq_len, c, h, w)
        features = self.qfilter(x_flat)  # (batch*seq_len, feature_dim)
        # Reshape back to a sequence for the LSTM
        features_seq = features.view(batch, seq_len, -1).transpose(0, 1)  # (seq_len, batch, feature_dim)
        lstm_out, _ = self.lstm(features_seq)
        logits = self.hidden2tag(lstm_out)  # (seq_len, batch, tagset_size)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuanvolutionQLSTM"]
