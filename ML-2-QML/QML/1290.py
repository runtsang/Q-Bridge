"""Quantum‑enhanced fully‑connected model with variational circuit and noise simulation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.noise import NoiseModel
import torchquantum.functional as tqf


class QFCModelExtended(tq.QuantumModule):
    """Variational quantum circuit with classical post‑processing for 4‑dimensional output."""

    class VariationalLayer(tq.QuantumModule):
        """Parameterized entangling circuit with multiple layers."""

        def __init__(self, n_wires: int, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # Create a list of RX, RY, RZ parameterised gates
            self.rx = [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            self.ry = [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            self.rz = [tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)]
            # Entangling layers
            self.cnot = [tq.CNOT(wires=[i, (i + 1) % n_wires]) for i in range(n_wires)]

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                # Apply single‑qubit rotations
                for i in range(self.n_wires):
                    self.rx[i](qdev, wires=i)
                    self.ry[i](qdev, wires=i)
                    self.rz[i](qdev, wires=i)
                # Apply entangling CNOTs
                for cnot_gate in self.cnot:
                    cnot_gate(qdev)

    def __init__(self, n_wires: int = 4, noise_model: NoiseModel | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.var_layer = self.VariationalLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_wires, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
        )
        self.noise_model = noise_model
        self.apply_noise = noise_model is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
            noise_model=self.noise_model if self.apply_noise else None,
        )
        # Global average pooling to 16 features per sample
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.var_layer(qdev)
        out = self.measure(qdev)  # shape: (bsz, n_wires)
        out = self.norm(out)
        out = self.classifier(out)
        return out

    def set_noise(self, noise_model: NoiseModel) -> None:
        """Attach a noise model to the device for subsequent forward passes."""
        self.noise_model = noise_model
        self.apply_noise = True

    def remove_noise(self) -> None:
        """Detach any noise model."""
        self.noise_model = None
        self.apply_noise = False


__all__ = ["QFCModelExtended"]
