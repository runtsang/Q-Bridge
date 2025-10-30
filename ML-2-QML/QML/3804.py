import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class Conv(tq.QuantumModule):
    """
    Quantum analogue of the Conv class.  It applies a quantum convolutional
    filter followed by a quantum fully‑connected head.  The implementation
    mirrors the classical architecture but replaces dense layers with
    parameter‑free quantum circuits.
    """
    def __init__(self, patch_size: int = 2, threshold: float = 0.5) -> None:
        super().__init__()
        # Quantum convolution block
        self.quantum_conv = self._QuantumConvolution(patch_size=patch_size, n_wires=patch_size**2, threshold=threshold)
        # Quantum fully‑connected head
        self.qfc = self._QuantumFullyConnected()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantum_conv(x)
        return self.qfc(x)

    class _QuantumConvolution(tq.QuantumModule):
        def __init__(self, patch_size: int = 2, n_wires: int = 4, threshold: float = 0.5) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.n_wires = n_wires
            self.threshold = threshold
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{patch_size}x{patch_size}_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, c, h, w = x.shape
            patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(-1, c, self.patch_size, self.patch_size)
            patch_vec = patches.view(-1, c * self.patch_size * self.patch_size)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=patch_vec.size(0), device=x.device)
            self.encoder(qdev, patch_vec)
            self.random_layer(qdev)
            out = self.measure(qdev)
            h_ = h // self.patch_size
            w_ = w // self.patch_size
            out = out.view(bsz, c, h_, w_, self.n_wires)
            return out

    class _QuantumFullyConnected(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, c, h, w = x.shape
            x_flat = x.view(bsz, -1)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            self.encoder(qdev, x_flat)
            self.random_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)

__all__ = ["Conv"]
