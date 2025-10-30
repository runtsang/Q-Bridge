"""Hybrid quantum-classical network combining quanvolution filter and quantum sampler."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter applying a random 2-qubit kernel to 2x2 image patches."""
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

class QuantumSamplerModule(nn.Module):
    """Quantum sampler network using Qiskit."""
    def __init__(self) -> None:
        super().__init__()
        self.inputs2 = ParameterVector("input", 2)
        self.weights2 = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(self.inputs2[0], 0)
        qc.ry(self.inputs2[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights2[0], 0)
        qc.ry(self.weights2[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights2[2], 0)
        qc.ry(self.weights2[3], 1)
        sampler = Sampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.inputs2,
            weight_params=self.weights2,
            sampler=sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 2)
        return self.sampler_qnn(inputs)

class QuanvolutionClassifier(nn.Module):
    """Quantum classifier head for quanvolution features."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class QuanvolutionSamplerNet(nn.Module):
    """Hybrid quantum-classical network: quanvolution filter → linear head → quantum sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.sampler = QuantumSamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        probs = F.log_softmax(logits, dim=-1)
        # Use first two logits as inputs to quantum sampler
        sampler_input = logits[:, :2]
        sampler_output = self.sampler(sampler_input)
        return torch.cat([probs, sampler_output], dim=-1)

__all__ = [
    "QuanvolutionFilter",
    "QuantumSamplerModule",
    "QuanvolutionClassifier",
    "QuanvolutionSamplerNet",
]
