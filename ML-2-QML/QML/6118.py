import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum quanvolution hybrid that maps 2×2 image patches to a
    4‑qubit variational circuit. The circuit parameters are trained
    jointly with the classical MLP head for richer feature extraction.
    """
    def __init__(self, patch_size=2, stride=2, mlp_hidden=128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_wires = 4  # 4 qubits per patch
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.var_layer = tq.ParameterizedLayer(
            n_layers=2,
            n_ops=2,
            op_type="ry",
            wires=list(range(self.n_wires)),
            entangling_fun=tq.entanglement,
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        n_patches = (28 // self.patch_size) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(self.n_wires * n_patches, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 10)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = x[:, r:r+self.patch_size, c:c+self.patch_size]
                data = patch.view(bsz, -1)
                self.encoder(qdev, data)
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        features = torch.cat(patches, dim=1)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)
