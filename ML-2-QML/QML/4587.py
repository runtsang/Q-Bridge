import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel based on a 4‑wire ansatz and a random layer."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.n_ops = n_ops
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=self.n_ops, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between batch `x` and `support` vectors."""
        bsz = x.shape[0]
        n_support = support.shape[0]
        out = []
        for sv in support:
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device, record_op=True)
            self.ansatz(qdev, x)
            self.random_layer(qdev)
            # encode negative support vector
            sv_expanded = sv.unsqueeze(0).expand(bsz, -1)
            self.ansatz(qdev, -sv_expanded)
            meas = self.measure(qdev)
            out.append(torch.abs(meas).unsqueeze(1))
        return torch.cat(out, dim=1)  # (bsz, n_support)

class HybridNAT(tq.QuantumModule):
    """Quantum‑enhanced variant of the hybrid model.

    Features a classical CNN feature extractor followed by a quantum kernel
    that maps the flattened representation to a similarity vector with
    learnable support vectors.  The similarity vector is then fed to a
    linear classifier.
    """
    def __init__(self,
                 n_support: int = 10,
                 n_wires: int = 4,
                 n_ops: int = 50,
                 n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 16 * 7 * 7
        self.proj = nn.Linear(self.flatten_dim, n_wires)
        self.kernel = QuantumKernel(n_wires=n_wires, n_ops=n_ops)
        self.support = nn.Parameter(torch.randn(n_support, n_wires))
        self.linear = nn.Linear(n_support, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x).view(bsz, -1)
        feat_proj = self.proj(feat)
        k = self.kernel(feat_proj, self.support)
        out = self.linear(k)
        return out

__all__ = ["HybridNAT"]
