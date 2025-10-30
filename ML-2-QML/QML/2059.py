import torch
import torch.nn as nn
import torchquantum as tq

class Encoder(tq.QuantumModule):
    """Applies a parameterized RY rotation to each qubit based on classical features."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.ry = tq.RY(has_params=True, trainable=False)

    def forward(self, qdev: tq.QuantumDevice, angles: torch.Tensor):
        # angles: (batch_size, n_wires)
        for wire in range(self.n_wires):
            self.ry(qdev, wires=wire, params=angles[:, wire])

class QFCModel(tq.QuantumModule):
    """Enhanced quantum fully connected model with a richer variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=100, wires=list(range(self.n_wires)))
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # First random layer
            self.random_layer(qdev)
            # Single‑qubit rotations
            for wire in range(self.n_wires):
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            # Entanglement chain
            for i in range(self.n_wires - 1):
                self.crx(qdev, wires=[i, i + 1])
            # Second random layer
            self.random_layer(qdev)
            # Final rotations
            for wire in range(self.n_wires):
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder_fc = nn.Linear(784, n_wires)
        self.encoder = Encoder(n_wires)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Flatten image to 784 features
        flat = x.view(bsz, -1)
        # Encode classical features into rotation angles
        angles = self.encoder_fc(flat)
        self.encoder(qdev, angles)
        # Variational circuit
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModel"]
