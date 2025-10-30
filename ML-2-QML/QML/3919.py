"""Quantum‑enhanced quanvolution with a parameterized sampler.

The quantum module mirrors the classical architecture while
leveraging a parameterized quantum circuit for patch encoding
and a quantum sampler that outputs a probability distribution.
It is built on top of TorchQuantum to remain fully
integrated with PyTorch autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.quantumdevice import QuantumDevice
from torchquantum.common import ParameterVector


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum patch encoder that processes non‑overlapping 2×2 image patches.
    Each patch is encoded into a 4‑qubit state, entangled via a random layer,
    and measured in the Pauli‑Z basis to produce four classical bits.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Simple Ry encoder for each pixel value
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = QuantumDevice(self.n_wires, bsz=bsz, device=device)
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
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (batch, 4*14*14)


class QuantumSampler(tq.QuantumModule):
    """
    Parameterized quantum sampler that maps the encoded features to a 2‑class
    probability distribution. It uses a small 2‑qubit circuit with Ry
    rotations and a CX gate, followed by a state‑vector sampler.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Construct a simple 2‑qubit circuit
        self.circuit = tq.Circuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        self.sampler = tq.StateVectorSampler(device=QuantumDevice(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2)
        # Map input features to the circuit's input parameters
        self.circuit.input_params[0] = x[:, 0]
        self.circuit.input_params[1] = x[:, 1]
        result = self.sampler(self.circuit)
        # Convert statevector to probabilities over |00>, |01>, |10>, |11>
        probs = result.probabilities
        # Sum over states to produce a 2‑dimensional output (e.g., |0> vs |1>)
        return probs[:, :2]  # (batch, 2)


class QuanvolutionSamplerNet(tq.QuantumModule):
    """
    Quantum‑enhanced network combining the quanvolution filter, quantum sampler,
    and a final classical classification head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        # feature_dim = 4 * 14 * 14
        self.feature_dim = 4 * 14 * 14
        self.sampler = QuantumSampler()
        self.classifier = nn.Linear(self.feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)          # (batch, feature_dim)
        sampler_out = self.sampler(features[:, :2])  # use first two dims for sampler
        logits = self.classifier(features)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, sampler_out


__all__ = ["QuanvolutionSamplerNet", "QuantumQuanvolutionFilter", "QuantumSampler"]
