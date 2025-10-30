import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import math

# Simple quantum encoder that maps a classical vector into a circuit
class QuantumEncoder(tq.QuantumModule):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.rxs = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.cnot = tq.CNOT

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        self.encoder(qdev, x)
        for i, gate in enumerate(self.rxs):
            gate(qdev, wires=i)
        for i in range(self.n_qubits - 1):
            self.cnot(qdev, wires=[i, i + 1])
        return tqf.measure_all(qdev, tq.PauliZ)

# Quantum‑enhanced classifier that embeds the CNN feature vector
class HybridNAT(tq.QuantumModule):
    """
    Quantum‑augmented hybrid model that mirrors the classical HybridNAT
    but replaces the transformer encoder with a variational quantum encoder.
    """
    def __init__(
        self,
        num_classes: int = 1,
        embed_dim: int = 64,
        patch_size: int = 4,
        n_qubits: int = 16,
    ) -> None:
        super().__init__()
        # Classical CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.patch_size = patch_size
        self.project = nn.Linear(32 * patch_size * patch_size, n_qubits)
        self.encoder = QuantumEncoder(n_qubits)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        features = self.cnn(x)
        b, c, h, w = features.shape
        # Flatten to a vector per example
        flat = features.view(b, -1)
        # Project to the number of qubits
        proj = self.project(flat)
        # Quantum encoding
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_qubits, bsz=b, device=proj.device)
        meas = self.encoder(qdev, proj)
        # Classical classification head
        return self.classifier(meas)
