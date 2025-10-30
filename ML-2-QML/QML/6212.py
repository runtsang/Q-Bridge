from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class QuanvolutionSamplerNet(tq.QuantumModule):
    """
    Quantum counterpart of QuanvolutionSamplerNet.  It applies a random
    two‑qubit quantum kernel to each 2×2 image patch (the quanvolution
    filter) and then feeds the compressed feature vector into a
    parameterised quantum sampler circuit that emits class probabilities.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4

        # Encoder mapping 4 classical patch values to qubit rotations
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

        # Linear layer to compress features into 2 inputs for the sampler
        self.feature_to_input = nn.Linear(4 * 14 * 14, 2)

        # Quantum sampler circuit
        input_params = ParameterVector("input", 2)
        weight_params = ParameterVector("weight", 4)
        qc = tq.QuantumCircuit(2)
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.cx(0, 1)
        for i in range(4):
            qc.ry(weight_params[i], i % 2)
        qc.cx(0, 1)
        self.sampler_circuit = qc

        self.sampler = SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=input_params,
            weight_params=weight_params,
            sampler=StatevectorSampler()
        )

        # Map sampler output (probabilities) to class logits
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Prepare patches
        patches = []
        x = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)

        # Compress to 2‑dimensional input for the sampler
        sampler_input = self.feature_to_input(features)
        probs = self.sampler(sampler_input)  # shape (bsz, num_classes)
        logits = self.classifier(probs)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionSamplerNet"]
