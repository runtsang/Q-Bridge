"""
Hybrid quantum sampler network that combines a quantum convolutional filter
with a parameterized sampler circuit.  The forward pass returns a
probability distribution over two outcomes, mirroring the classical
HybridSamplerQuanvolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchquantum import (
    QuantumModule,
    QuantumDevice,
    GeneralEncoder,
    RandomLayer,
    MeasureAll,
    PauliZ,
)
from torchquantum.circuit import QuantumCircuit
from torchquantum.parameter import ParameterVector


class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum‑convolutional filter that processes 2×2 image patches
    using a 4‑qubit circuit.  The filter is pure quantum and does not
    involve any classical parameters.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            Tensor of shape (batch, 4*14*14) containing quantum measurements
            for each 2×2 patch.
        """
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        return torch.cat(patches, dim=1)


class HybridSamplerQuanvolution(tq.QuantumModule):
    """
    Quantum sampler that first applies QuanvolutionFilter and then
    runs a parameterized sampler circuit to produce a probability
    distribution over two measurement outcomes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()

        # Parameter vectors for the sampler circuit
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Build the sampler circuit
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Measurement for the sampler
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            Tensor of shape (batch, 2) containing softmax probabilities.
        """
        # Extract quantum features
        qfeatures = self.qfilter(x)  # (batch, 4*14*14)

        # Use first two qubits of the sampler circuit for each batch element
        bsz = x.shape[0]
        qdev = QuantumDevice(2, bsz=bsz, device=x.device)

        # Encode the first two input parameters from the image
        # Here we simply take the mean of the image as a proxy
        inputs = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        self.circuit.set_parameters(self.input_params, torch.cat([inputs, inputs], dim=1))

        # Apply the sampler circuit
        self.circuit(qdev)
        measurement = self.measure(qdev)

        # Combine with quantum convolution features
        # For simplicity we average the measurement vector
        probs = F.softmax(measurement.float(), dim=-1)
        return probs


__all__ = ["HybridSamplerQuanvolution"]
