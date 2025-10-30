"""Quantum‑enhanced LSTM with fraud‑detection style preprocessor.

The QLSTM uses parameterised quantum circuits for the gates.
A quantum preprocessor built with TorchQuantum emulates the
photonic fraud‑detection layer.  The LSTMTagger stitches
the embedding, optional quantum preprocessor and the quantum
LSTM together for sequence tagging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
#  Fraud‑preprocessing utilities (quantum)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameter set for a fraud‑detection style quantum block."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudQuantumPreprocessor(tq.QuantumModule):
    """Quantum preprocessor that maps a 2‑dim input to a 2‑dim output
    via a parameterised circuit inspired by the photonic fraud‑detection
    layer.  Parameters are taken from a :class:`FraudLayerParameters`
    instance and encoded as rotation angles.
    """

    def __init__(self, params: FraudLayerParameters) -> None:
        super().__init__()
        self.params = params
        self.n_wires = 2
        # Parameterised gates
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.rx1 = tq.RX(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.rz1 = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        # Encode raw inputs
        self.rx0(qdev, x[:, 0], wires=0)
        self.rx1(qdev, x[:, 1], wires=1)
        # Apply fraud‑layer parameters as rotation angles
        self.rx0(qdev, torch.tensor(self.params.bs_theta, device=x.device), wires=0)
        self.rx1(qdev, torch.tensor(self.params.bs_phi, device=x.device), wires=1)
        self.rz0(qdev, torch.tensor(self.params.phases[0], device=x.device), wires=0)
        self.rz1(qdev, torch.tensor(self.params.phases[1], device=x.device), wires=1)
        # Entangle the two wires
        self.cnot(qdev, wires=[0, 1])
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
#  Quantum LSTM implementation
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """LSTM cell where each gate is implemented by a small quantum circuit."""

    class QGate(tq.QuantumModule):
        """A reusable quantum gate block for an LSTM gate."""

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Simple parameterised circuit
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.cnot = tq.CNOT()
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.rx0(qdev, x[:, 0], wires=0)
            self.rx1(qdev, x[:, 1], wires=1)
            self.cnot(qdev, wires=[0, 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_wires = n_qubits

        # Linear maps that convert classical concatenation to n_qubits‑dim vectors
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gate blocks
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM
    and optionally prepend a quantum fraud‑detection preprocessor.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        fraud_params: Optional[FraudLayerParameters] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.preprocessor = None
        if fraud_params is not None:
            self.preprocessor = FraudQuantumPreprocessor(fraud_params)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        if self.preprocessor is not None:
            embeds = self.preprocessor(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger", "FraudLayerParameters", "FraudQuantumPreprocessor"]
