"""Quantum implementation of the HybridQuantumNAT model.

Uses TorchQuantum for a quantum fully-connected layer, StrawberryFields for a photonic fraud-detection block, and a classical estimator head.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate, Zgate

class HybridQuantumNAT(tq.QuantumModule):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_dim = 16 * 7 * 7

        # Quantum fully connected layer
        self.qfc = _QuantumFullyConnectedLayer(n_wires=4)

        # Photonic fraud-detection layer
        self.fraud = _PhotonicFraudLayer()

        # Classical estimator head
        self.estimator = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)

        # Quantum encoding of features into qubits
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=self.device, record_op=True)
        self.qfc.encode(qdev, flattened)
        out = self.qfc(qdev)

        # Photonic fraud detection
        out = self.fraud(out)

        # Estimator head
        out = self.estimator(out)
        return out

class _QuantumFullyConnectedLayer(tq.QuantumModule):
    """Quantum fully connected layer using TorchQuantum."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def encode(self, qdev: tq.QuantumDevice, features: torch.Tensor) -> None:
        # simple encoding: map first 4 feature components to Ry rotations
        for i in range(self.n_wires):
            tqf.ry(qdev, features[:, i], wires=[i])

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])
        out = self.measure(qdev)
        return self.norm(out)

class _PhotonicFraudLayer(nn.Module):
    """Photonic fraud detection block using StrawberryFields."""
    def __init__(self):
        super().__init__()
        self.backend = sf.backends.FockBackend(cutoff_dim=5)
        self.program = sf.Program(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz = inputs.shape[0]
        out = torch.zeros(bsz, 2, device=inputs.device, dtype=torch.float32)
        for i in range(bsz):
            amp0 = inputs[i, 0].item()
            amp1 = inputs[i, 1].item()
            with self.program.context as q:
                Dgate(amp0, 0) | q[0]
                Dgate(amp1, 0) | q[1]
                BSgate(0.5, 0) | (q[0], q[1])
            state = self.backend.run(self.program).state
            # Expectation of Pauli Z on each mode (placeholder)
            exp0 = state.expectation_value(Zgate)
            exp1 = state.expectation_value(Zgate)
            out[i, 0] = exp0
            out[i, 1] = exp1
        return out

__all__ = ["HybridQuantumNAT"]
