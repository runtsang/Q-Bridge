import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum filter that processes 2x2 image patches with a random layer."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Quantum layer that maps a 4‑dim input to 4 outputs."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.param_rx = tq.RX(has_params=True)
        self.param_ry = tq.RY(has_params=True)
        self.param_rz = tq.RZ(has_params=True)
        self.param_crx = tq.CRX(has_params=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        self.param_rx(qdev, wires=0)
        self.param_ry(qdev, wires=1)
        self.param_rz(qdev, wires=3)
        self.param_crx(qdev, wires=[0, 2])
        measurement = self.measure(qdev)
        out = self.norm(measurement)
        return out

class QuanvolutionHybridModel(tq.QuantumModule):
    """Hybrid quantum‑classical model that combines a quantum patch filter,
    a quantum fully connected layer, and a classical linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.qfc = QuantumFullyConnectedLayer()
        self.head = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.qfilter(x)  # (bsz, 784)
        patches_avg = patches.view(x.shape[0], 14, 14, 4).mean(dim=(1, 2))  # (bsz, 4)
        qfc_out = self.qfc(patches_avg)  # (bsz, 4)
        logits = self.head(qfc_out)      # (bsz, 10)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridModel"]
