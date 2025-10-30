"""QML module for hybrid quantum regression.

Implements a quantum encoder, a quanvolution filter, a quantum sampler head and
an integrated hybrid model that fuses classical and quantum features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

# --------------------------------------------------------------------------- #
#  Quantum encoder – maps classical inputs into a quantum state
# --------------------------------------------------------------------------- #
class QuantumEncoder(tq.QuantumModule):
    """Encodes a 1‑D feature vector into a quantum state using Ry gates."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.enc = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(num_wires)
            ]
        )

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.enc(qdev)

# --------------------------------------------------------------------------- #
#  Quanvolution filter – 2×2 patch embedding via a small quantum kernel
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
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

# --------------------------------------------------------------------------- #
#  Quantum sampler – lightweight QNN for probability distribution
# --------------------------------------------------------------------------- #
class SamplerQNN(tq.QuantumModule):
    """A parameterised quantum circuit that outputs a probability vector."""
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        # Parameter vectors for inputs and trainable weights
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build a simple circuit
        self.circuit = tq.QuantumCircuit(self.n_wires)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

        # Sampler primitive
        self.sampler = Sampler()
        self.qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Convert the tq.QuantumDevice into a statevector that qiskit can sample
        state = qdev._statevectors
        # Here we simply return the probabilities from the qiskit sampler
        probs = self.qnn(state)
        return probs  # shape (bsz, 4)

# --------------------------------------------------------------------------- #
#  Hybrid model – fuses classical and quantum branches
# --------------------------------------------------------------------------- #
class QuantumRegressionHybrid(tq.QuantumModule):
    """Hybrid regression model with classical MLP, quantum encoder, quanvolution,
    sampler head and a shared linear output."""
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QuantumEncoder(num_wires)

        # Classical branch
        self.classical_head = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # Quantum branch
        self.qfilter = QuanvolutionFilter()
        self.qsampler = SamplerQNN()

        # Fusion head
        # Number of features: classical (8) + quanvolution (4*14*14) + sampler (4)
        self.fusion_head = nn.Linear(8 + 4 * 14 * 14 + 4, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        device = state_batch.device

        # Classical feature extraction
        classical_feats = self.classical_head(state_batch)

        # Quantum encoding
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev)
        # Apply quanvolution on a reshaped patch grid
        quanv_feats = self.qfilter(state_batch.unsqueeze(1))  # add channel dim

        # Sampler output
        sampler_feats = self.qsampler(qdev)

        # Concatenate all modalities
        fused = torch.cat([classical_feats, quanv_feats, sampler_feats], dim=1)
        return self.fusion_head(fused).squeeze(-1)

__all__ = [
    "QuantumEncoder",
    "QuanvolutionFilter",
    "SamplerQNN",
    "QuantumRegressionHybrid",
]
