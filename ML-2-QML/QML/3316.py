import torch
import torch.nn as nn
import torchquantum as tq

class QuantumHybridNAT(tq.QuantumModule):
    """
    Quantum-only model that processes 28x28 images patch-wise using a quantum kernel and outputs class logits.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.n_wires = 4
        # Encoder for each patch: 2x2 patch encoded into 4 qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        bsz = x.shape[0]
        device = x.device
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]
                patch = patch.view(bsz, -1)
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement)
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return self.norm(logits)
