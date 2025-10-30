'''python
# Quantum‑classical hybrid model that applies a quantum kernel to image patches
# and classifies with a SamplerQNN.  The quantum kernel encodes each 2×2 patch
# into a 4‑qubit state using Ry rotations, measures all qubits, and then
# reduces the resulting 784‑dimensional feature vector to 2 dimensions
# before the SamplerQNN performs probabilistic classification.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class QuanvolutionHybrid(tq.QuantumModule):
    def __init__(self, num_classes: int = 10, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Quantum kernel ansatz: 4‑qubit Ry rotations encoding the patch
        self.ansatz = lambda q, x: [
            tq.ry(q, wires=[i], params=x[:, i]) for i in range(self.n_wires)
        ]

        # Classical linear layer to reduce the 784‑dimensional patch feature vector
        # to 2 dimensions for the SamplerQNN input.
        self.fc = nn.Linear(784, 2)

        # SamplerQNN classifier
        self.qnn = self._build_sampler_qnn(num_classes)

    def _build_sampler_qnn(self, num_classes: int) -> SamplerQNN:
        input_params = ParameterVector("input", 2)
        weight_params = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)

        sampler = Sampler()
        return SamplerQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            sampler=sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz = x.shape[0]
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r : r + 2, c : c + 2].reshape(bsz, -1)  # (batch, 4)
                self.q_device.reset_states(bsz)
                self.ansatz(self.q_device, patch)
                meas = self.q_device.measure_all()  # (batch, 2**n_wires)
                patches.append(meas.view(bsz, -1))

        # Concatenate all patch features: (batch, 784)
        features = torch.cat(patches, dim=1)

        # Reduce dimensionality for the SamplerQNN
        reduced = self.fc(features)  # (batch, 2)

        # Classification with the quantum neural network
        logits = self.qnn(reduced)  # (batch, num_classes)

        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
'''
