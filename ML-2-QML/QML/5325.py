"""Hybrid QML implementation of a Quantum LSTM with optional fraud detection,
convolutional filter and estimator QNN.

This module defines a `HybridQLSTM` class that can operate in purely quantum
mode or fall back to classical LSTM when `n_qubits=0`.  All auxiliary
components are implemented with TorchQuantum or Qiskit to showcase
quantum‑enhanced feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Fraud‑Detection layer (quantum)
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QuantumFraudLayer(tq.QuantumModule):
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        self.params = params
        self.clip = clip
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
            ]
        )
        self.params_list = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params_list:
            gate(qdev)
        return self.measure(qdev)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [QuantumFraudLayer(input_params, clip=False)]
    modules.extend(QuantumFraudLayer(layer, clip=True) for layer in layers)
    return nn.Sequential(*modules)

# Quantum convolutional filter
def Conv() -> tq.QuantumModule:
    """Return a simple quantum filter using a random circuit."""
    class QuanvCircuit(tq.QuantumModule):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.n_qubits = kernel_size ** 2
            self.threshold = threshold
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(self.n_qubits)
                ]
            )
            self.random_circuit = tq.RandomCircuit(self.n_qubits, depth=2)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            self.random_circuit(qdev)
            return self.measure(qdev)

    return QuanvCircuit()

# Estimator QNN (quantum)
def EstimatorQNN() -> tq.QuantumModule:
    """Quantum estimator using a simple parameter‑shift circuit."""
    class EstimatorNN(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_qubits = 1
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]}]
            )
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            self.rz(qdev)
            return self.measure(qdev)

    return EstimatorNN()

# Quantum LSTM cell
class QuantumQLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
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
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
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
        return torch.cat(outputs, dim=0), (hx, cx)

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

# Classical fallback LSTM
class ClassicalQLSTM(nn.Module):
    """Classic LSTM implementation used when n_qubits == 0."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs, (hx, cx) = self.lstm(inputs, states)
        return outputs, (hx, cx)

# Hybrid model
class HybridQLSTM(nn.Module):
    """Sequence tagging model that can mix classical and quantum LSTM layers
    and optionally apply quantum fraud‑detection, convolution and estimator heads.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_fraud: bool = False,
        use_conv: bool = False,
        use_estimator: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional quantum components
        self.use_fraud = use_fraud
        self.fraud_preprocessor = None
        if use_fraud:
            dummy_params = FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.5, phases=(0.1, 0.2),
                squeeze_r=(0.3, 0.3), squeeze_phi=(0.1, 0.1),
                displacement_r=(0.2, 0.2), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            self.fraud_preprocessor = build_fraud_detection_program(dummy_params, [])

        self.use_conv = use_conv
        self.conv_filter = None
        if use_conv:
            self.conv_filter = Conv()

        self.use_estimator = use_estimator
        self.estimator_head = None
        if use_estimator:
            self.estimator_head = EstimatorQNN()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Optional fraud pre‑processing
        if self.use_fraud and self.fraud_preprocessor is not None:
            flat = self.word_embeddings(sentence).view(-1, 2)
            processed = self.fraud_preprocessor(flat)
            processed = processed.view(sentence.size(0), -1, self.hidden_dim)
            lstm_input = processed
        else:
            lstm_input = self.word_embeddings(sentence).view(sentence.size(0), 1, -1)

        lstm_out, _ = self.lstm(lstm_input)
        tag_logits = self.hidden2tag(lstm_out.view(sentence.size(0), -1))

        # Optional estimator head
        if self.use_estimator and self.estimator_head is not None:
            est_out = self.estimator_head(lstm_out.view(-1, self.hidden_dim))
            est_out = est_out.view(sentence.size(0), -1)
            tag_logits = torch.cat([tag_logits, est_out], dim=-1)

        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM"]
