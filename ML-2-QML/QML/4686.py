import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Sequence

class QuantumQFCModel(tq.QuantumModule):
    """Quantum feature extractor mirroring the classical CNN architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=32, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.random_layer(qdev)
        out = self.measure(qdev)
        return out

class QuantumKernel(tq.QuantumModule):
    """Variational quantum kernel built on top of the quantum feature extractor."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.qf_model = QuantumQFCModel()
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feat = self.qf_model(x)
        y_feat = self.qf_model(y)
        bsz = x_feat.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        qdev.reset_states()
        # Encode x (positive angles)
        for i in range(self.n_wires):
            tq.RY(qdev, wires=[i], params=x_feat[:, i])
        # Entangle
        for i in range(self.n_wires - 1):
            tq.CX(qdev, wires=[i, i + 1])
        # Encode y (negative angles)
        for i in range(self.n_wires):
            tq.RY(qdev, wires=[i], params=-y_feat[:, i])
        out = self.measure(qdev)
        return torch.abs(out[:, 0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  shots: int | None = None, seed: int | None = None) -> np.ndarray:
    """Compute the quantum kernel Gram matrix with optional shot noise."""
    kernel = QuantumKernel()
    mat = np.array([[kernel(x, y).item() for y in b] for x in a])
    if shots is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1.0/np.sqrt(shots), size=mat.shape)
        mat += noise
    return mat

__all__ = ["QuantumQFCModel", "QuantumKernel", "kernel_matrix"]
