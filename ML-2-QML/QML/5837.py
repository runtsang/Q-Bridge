import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Extended quantum model with a variational ansatz and a POVM measurement.
    """
    class Entangler(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Random layer for feature encoding
            self.encoder = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Two‑qubit entangling block
            self.cnot = tq.CNOT
            # Parameterized single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.encoder(qdev)
            # Entangle pairs in a ladder pattern
            for i in range(self.n_wires - 1):
                self.cnot(qdev, wires=[i, i + 1])
            # Apply rotations
            for i in range(self.n_wires):
                self.rx(qdev, wires=i)
                self.ry(qdev, wires=i)
                self.rz(qdev, wires=i)

    def __init__(self, n_wires: int = 4, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.qlayer = self.Entangler(n_wires)
        # POVM layer: computational‑basis POVM returning probabilities
        self.measure = tq.POVM(povm_ops=[tq.PauliZ for _ in range(n_wires)])
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Simple feature pooling: average over spatial dims
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, self.n_wires)
        # Encode features into qubits
        self.qlayer(qdev)
        out = self.measure(qdev)  # returns probability distribution
        return self.norm(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
        return logits

    def score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        logits = self.predict(x)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()
