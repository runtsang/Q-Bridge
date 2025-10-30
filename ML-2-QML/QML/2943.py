"""Hybrid quantum‑classical network that uses a quanvolution filter and a variational sampler."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit


class QuanvolutionFilterQ(tq.QuantumModule):
    """Quantum quanvolution filter that applies a random two‑qubit layer to 2×2 patches."""
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


class QuanvolutionSamplerNet(nn.Module):
    """
    Quantum hybrid network:
        quanvolution filter -> linear -> quantum sampler -> linear -> log‑softmax
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterQ()
        self.reduce = nn.Linear(4 * 14 * 14, 2)

        # Define parameterized quantum circuit for the sampler
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
            circuit=qc2,
            input_params=inputs2,
            weight_params=weights2,
            sampler=sampler,
        )

        self.head = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        x = self.reduce(x)
        x = self.sampler_qnn(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionSamplerNet"]
