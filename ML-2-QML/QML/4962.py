"""Quantum‑NAT variant that replaces the dense head with a variational
quantum circuit.  The model re‑uses the CNN backbone from the classical
variant, encodes the flattened features into a quantum device, applies a
random layer followed by trainable rotation gates, measures all qubits,
and maps the expectation values to the final output via a linear head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class HybridFunction(tq.QuantumModule):
    """Differentiable interface that propagates gradients through the quantum
    circuit by using the parameter-shift rule."""
    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:  # type: ignore[override]
        # Random layer
        self.random_layer(qdev)
        # Trainable rotations
        for wire in range(self.n_wires):
            self.rx(qdev, wire)
            self.ry(qdev, wire)
        # Classical post‑processing
        return tqf.hadamard(qdev, wires=self.n_wires - 1, static=self.static_mode, parent_graph=self.graph)


class QLayer(tq.QuantumModule):
    """Variational block that will be applied after encoding."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wire)
            self.ry(qdev, wire)


class QuantumNATGen191(tq.QuantumModule):
    """Hybrid CNN + quantum circuit.  The quantum block replaces the
    dense head from the classical variant."""
    def __init__(self, num_input_channels: int = 1, n_wires: int = 4,
                 use_sigmoid: bool = True, shift: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Classical feature extractor identical to the ML version
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        # Encoder that maps classical features into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4xRy"])
        self.quantum_layer = QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)
        # Classical linear head that maps quantum expectations to logits
        self.head = nn.Linear(n_wires, 1) if use_sigmoid else nn.Linear(n_wires, 1)
        self.use_sigmoid = use_sigmoid
        self.shift = shift

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical feature extraction
        feat = self.features(x)
        flattened = self.flatten(feat)
        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode the flattened features
        self.encoder(qdev, flattened)
        # Variational block
        self.quantum_layer(qdev)
        # Expectation values
        out = self.measure(qdev)
        out = self.norm(out)
        # Map to logits
        logits = self.head(out)
        if self.use_sigmoid:
            logits = torch.sigmoid(logits + self.shift)
            return torch.cat((logits, 1 - logits), dim=-1)
        return logits.squeeze(-1)


class SamplerQNN(tq.QuantumModule):
    """Small parameterised sampler circuit used for generative purposes."""
    def __init__(self, n_qubits: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.ry = tq.RY(has_params=True, trainable=True)
        self.cx = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_qubits, 2)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.ry(qdev, 0)
        self.ry(qdev, 1)
        self.cx(qdev, [0, 1])
        self.measure(qdev)
        out = self.head(qdev)
        return torch.softmax(out, dim=-1)


__all__ = ["QuantumNATGen191", "SamplerQNN", "HybridFunction", "QLayer"]
