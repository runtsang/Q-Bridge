"""
Quantum components for HybridSamplerRegressor.
Combines a qiskit SamplerQNN with a quanvolution filter based on torchquantum.
"""

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as SamplerPrimitive

import torchquantum as tq
import torch.nn as nn
import torch.nn.functional as F

def SamplerQNN() -> QiskitSamplerQNN:
    """
    Parameterized quantum circuit for probabilistic sampling.
    Adds a small random layer and uses a StatevectorSampler backend
    for differentiable execution.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    # Input encoding
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)

    # Parameterized rotations
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = SamplerPrimitive()
    return QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )

class QuanvolutionFilter(tq.QuantumModule):
    """
    Apply a random two‑qubit quantum kernel to 2×2 image patches.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encode 4 classical inputs into 4 qubits using Ry gates
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, 28, 28) – grayscale images.
        Returns
        -------
        torch.Tensor
            Concatenated quantum measurements for all 2×2 patches.
            Shape (batch, 4 * 14 * 14)
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to (batch, 28, 28) and extract 2×2 patches
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
                ).to(device)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

__all__ = ["SamplerQNN", "QuanvolutionFilter"]
