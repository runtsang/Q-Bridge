import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

@dataclass
class FraudLayerParameters:
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

class FraudNATHybridModel(nn.Module):
    """Hybrid Fraud‑NAT model:
    1. CNN feature extractor
    2. Quantum circuit (torchquantum) producing 4‑dimensional output
    3. Linear classifier
    """

    def __init__(
        self,
        n_wires: int = 4,
        encoder_name: str = "4x4_ryzxy",
        fraud_params: Optional[FraudLayerParameters] = None,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Quantum encoder and layer
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.quantum_layer = self._build_quantum_layer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

        # Final classifier
        self.classifier = nn.Linear(n_wires, 1)

        # Optional photonic parameters for later use
        self.fraud_params = fraud_params

    def _build_quantum_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
                self.rx0 = tq.RX(has_params=True, trainable=True)
                self.ry0 = tq.RY(has_params=True, trainable=True)
                self.rz0 = tq.RZ(has_params=True, trainable=True)
                self.crx0 = tq.CRX(has_params=True, trainable=True)

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice) -> None:
                self.random_layer(qdev)
                self.rx0(qdev, wires=0)
                self.ry0(qdev, wires=1)
                self.rz0(qdev, wires=3)
                self.crx0(qdev, wires=[0, 2])
                tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
                tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
                tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        pooled = self.pool(feats).view(bsz, -1)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, pooled)
        self.quantum_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        out = self.classifier(out)
        return out

# Photonic fraud‑detection utility (kept for compatibility with the original seed)
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) ->'sf.Program':
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | modes[i]

    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

__all__ = ["FraudLayerParameters", "FraudNATHybridModel", "build_fraud_detection_program"]
