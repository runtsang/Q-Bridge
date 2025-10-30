"""Hybrid quantum–classical sampler using torchquantum and Qiskit."""
from __future__ import annotations

import torch
import torchquantum as tq
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class QuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""
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


class HybridSamplerQNN(tq.QuantumModule):
    """
    Quantum hybrid sampler: quanvolution feature extraction followed by a
    Qiskit SamplerQNN that produces a two‑class probability distribution.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()

        # Build the parameterized sampler circuit
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)

        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=sampler
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract quantum features with the quanvolution filter
        features = self.qfilter(x)  # shape: (bsz, 4*14*14)
        # Use the first two features as inputs to the sampler circuit
        inputs = features[:, :2].cpu().numpy()
        # Run the sampler QNN
        probs = self.sampler_qnn.forward(inputs)
        # Convert back to torch tensor
        return torch.from_numpy(probs).to(x.device)


__all__ = ["HybridSamplerQNN"]
