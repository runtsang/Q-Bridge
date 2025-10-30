import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import List

@dataclass
class FraudParams:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class HybridNATModel(tq.QuantumModule):
    """Quantum‑centred hybrid model with photonic‑style gates."""
    class QLayer(tq.QuantumModule):
        def __init__(self, fraud_params: List[FraudParams]):
            super().__init__()
            self.params = fraud_params
            self.n_wires = 4
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Random initialisation
            self.random(qdev)
            # Apply photonic‑style gates
            for idx, p in enumerate(self.params):
                # Beam Splitter
                tq.BSgate(p.bs_theta, p.bs_phi, wires=[0, 1])(qdev)
                for i, phase in enumerate(p.phases):
                    tq.Rgate(phase, wires=[i])(qdev)
                for i, (r, phi) in enumerate(zip(p.squeeze_r, p.squeeze_phi)):
                    tq.Sgate(r if not idx else _clip(r, 5), phi, wires=[i])(qdev)
                tq.BSgate(p.bs_theta, p.bs_phi, wires=[0, 1])(qdev)
                for i, phase in enumerate(p.phases):
                    tq.Rgate(phase, wires=[i])(qdev)
                for i, (r, phi) in enumerate(zip(p.displacement_r, p.displacement_phi)):
                    tq.Dgate(r if not idx else _clip(r, 5), phi, wires=[i])(qdev)
                for i, k in enumerate(p.kerr):
                    tq.Kgate(k if not idx else _clip(k, 1), wires=[i])(qdev)

    def __init__(self, fraud_params: List[FraudParams] | None = None):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(fraud_params or [])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Classical CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        self.head = nn.Sequential(
            nn.Linear(16 * 7 * 7 + self.n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        features = self.cnn(x)
        flat = features.view(bsz, -1)

        # Quantum encoding and evolution
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        q_out = self.measure(qdev)

        combined = torch.cat([flat, q_out], dim=1)
        out = self.head(combined)
        return self.norm(out)

__all__ = ["HybridNATModel", "FraudParams"]
