import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModelExtended(tq.QuantumModule):
    """Hybrid quantum model with measurement‑guided variational layers."""
    class VariationalLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Linear map from measurement to rotation angles
            self.angle_map = nn.Linear(n_wires, n_wires)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            # First variational block
            self.rx(qdev, wires=range(self.n_wires))
            self.ry(qdev, wires=range(self.n_wires))
            self.rz(qdev, wires=range(self.n_wires))
            tq.CNOT(qdev, wires=[0, 1], static=True)
            tq.CNOT(qdev, wires=[1, 2], static=True)
            tq.CNOT(qdev, wires=[2, 3], static=True)
            # Measurement
            meas = tq.MeasureAll(tq.PauliZ)(qdev)
            # Map measurement to rotation angles
            angles = self.angle_map(meas)
            # Apply feedback rotations
            self.rx(qdev, wires=range(self.n_wires), params=angles)
            # Second variational block
            self.ry(qdev, wires=range(self.n_wires))
            self.rz(qdev, wires=range(self.n_wires))
            tq.CNOT(qdev, wires=[0, 2], static=True)
            tq.CNOT(qdev, wires=[1, 3], static=True)
            # Final measurement
            return tq.MeasureAll(tq.PauliZ)(qdev)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.variational = self.VariationalLayer(self.n_wires)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        out = self.variational(qdev)
        return self.norm(out)


__all__ = ["QFCModelExtended"]
