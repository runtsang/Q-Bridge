import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATHybrid(nn.Module):
    """
    Hybrid classical‑quantum model integrating a CNN feature extractor,
    a fully‑connected projection, and a trainable quantum sub‑module.
    Supports batched inference on any device.
    """
    class QLayer(tq.QuantumModule):
        """Variational quantum layer built on random gates and trainable rotations."""
        def __init__(self, n_wires: int = 4, n_ops: int = 50):
            super().__init__()
            self.n_wires = n_wires
            # Random layer provides a dense entangling substrate
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Additional entangling gate
            self.cnot = tq.CNOT(has_params=False, trainable=False)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # First apply randomness
            self.random_layer(qdev)
            # Apply a small, learnable rotation pattern
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            # Entangle a pair of qubits
            self.cnot(qdev, wires=[0, 3])
            # Final Hadamard for basis change
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Classical front‑end: two conv layers + avg pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected mapping to match quantum register size
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_wires)
        )
        # Quantum module
        self.q_layer = self.QLayer(n_wires=n_wires)
        # Measurement and post‑processing
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features, encode into qubits, run quantum layer,
        and normalize outputs.
        """
        bsz = x.shape[0]
        feats = self.features(x)
        pooled = F.avg_pool2d(feats, kernel_size=6).view(bsz, -1)
        # Map to quantum register
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.fc(qdev, pooled)  # embed classical vector into qubits
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
