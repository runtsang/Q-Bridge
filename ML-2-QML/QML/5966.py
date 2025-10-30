import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["QuantumClassifierModel"]

class QuantumClassifierModel(tq.QuantumModule):
    """
    Quantum classifier that encodes classical features into a 2‑qubit
    variational circuit and returns the Z‑expectation values.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 2):
            super().__init__()
            self.n_wires = n_wires
            # Encoding gates (non‑trainable) to map input angles
            self.ry_enc = tq.RY(has_params=True, trainable=False)
            self.rz_enc = tq.RZ(has_params=True, trainable=False)
            # Variational block – trainable parameters
            self.ry_var = tq.RY(has_params=True, trainable=True)
            self.rz_var = tq.RZ(has_params=True, trainable=True)
            self.cz_var = tq.CZ()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, angles: torch.Tensor) -> None:
            # angles shape: (batch, 2 * n_wires)
            for i in range(self.n_wires):
                self.ry_enc(qdev, wires=i, params=angles[:, 2 * i])
                self.rz_enc(qdev, wires=i, params=angles[:, 2 * i + 1])
            # Variational block
            for i in range(self.n_wires):
                self.ry_var(qdev, wires=i)
                self.rz_var(qdev, wires=i)
            self.cz_var(qdev, wires=[0, 1])

    def __init__(self, num_features: int = 784, n_wires: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Linear(num_features, 2 * n_wires)
        self.n_wires = n_wires
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        angles = self.encoder(x)
        self.q_layer(qdev, angles)
        out = self.measure(qdev)
        return self.norm(out)
