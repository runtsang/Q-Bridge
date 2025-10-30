import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QuantumPhotonicLayer(tq.QuantumModule):
    """Small quantum circuit that mimics the photonic operations using
    rotation and two‑qubit gates."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
        self.params = params
        self.clip = clip
        self.n_wires = 2
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        theta = _clip(self.params.bs_theta, 5.0) if self.clip else self.params.bs_theta
        phi = _clip(self.params.bs_phi, 5.0) if self.clip else self.params.bs_phi
        self.random_layer(qdev)
        self.rx(qdev, wires=0, params=theta)
        self.ry(qdev, wires=1, params=phi)
        self.rz(qdev, wires=0, params=phi)
        self.crx(qdev, wires=[0, 1], params=theta)
        tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)


class FraudDetectionHybridQML(tq.QuantumModule):
    """Quantum‑classical hybrid fraud‑detection model that combines a CNN encoder
    with a stack of quantum‑photonic layers."""
    def __init__(self, *layer_params: FraudLayerParameters):
        super().__init__()
        self.n_wires = 4
        # Classical encoder (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Quantum photonic layers stack
        self.q_layers = nn.ModuleList(
            [QuantumPhotonicLayer(layer_params[0], clip=False)] +
            [QuantumPhotonicLayer(p, clip=True) for p in layer_params[1:]]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        pooled = self.pool(feats).view(bsz, -1)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        for layer in self.q_layers:
            layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["FraudDetectionHybridQML", "FraudLayerParameters"]
