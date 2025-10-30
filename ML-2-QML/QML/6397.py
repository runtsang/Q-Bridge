import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

def create_sampler_qnn() -> QiskitSamplerQNN:
    """Builds a simple 2‑qubit parameterised sampler."""
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
    sampler = Sampler()
    return QiskitSamplerQNN(
        circuit=qc2,
        input_params=inputs2,
        weight_params=weights2,
        sampler=sampler,
    )

class HybridQuanvolutionFilterQuantum(nn.Module):
    """Quantum 2×2 patch encoder with a random layer and sampler augmentation."""
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
        self.sampler_qnn = create_sampler_qnn()
        self.sampler = self.sampler_qnn.sampler

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
        quantum_features = torch.cat(patches, dim=1)
        # Sampler features: evaluate on zero input for each batch element
        dummy_inputs = np.zeros((bsz, 2))
        sampler_probs = self.sampler_qnn.sample(inputs=dummy_inputs)
        sampler_tensor = torch.tensor(sampler_probs, device=device, dtype=torch.float32)
        return torch.cat([quantum_features, sampler_tensor], dim=1)

class HybridQuanvolutionClassifierQuantum(nn.Module):
    """Quantum‑augmented classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = HybridQuanvolutionFilterQuantum()
        self.linear = nn.Linear(4 * 14 * 14 + 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionFilterQuantum", "HybridQuanvolutionClassifierQuantum"]
