import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumResidualBlock(tq.QuantumModule):
    """A simple quantum residual block that adds its output to the input tensor."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # Flatten input to 4 features per patch
        data = x.view(bsz, 4)
        self.encoder(qdev, data)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        # Add residual: input + measurement
        return x + measurement.view(bsz, 4)

class QuanvolutionFilter224Quantum(tq.QuantumModule):
    """Quantum quanvolution filter for 224×224 images using 3×3 patches and 4 qubits per patch."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.patch_size = 3
        self.stride = 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.residual = QuantumResidualBlock(n_wires=n_wires, n_ops=n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 224, 224)
        patches = []
        for r in range(0, 224, self.stride):
            for c in range(0, 224, self.stride):
                # Extract 3×3 patch
                patch = x[:, r:r+self.patch_size, c:c+self.patch_size]
                # Flatten and select first 4 elements for encoding
                data = patch.view(bsz, -1)[:, :4]
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                # Apply residual block
                res = self.residual(measurement.view(bsz, 4))
                patches.append(res)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifierQuantum(nn.Module):
    """Hybrid quantum–classical classifier using the 224×224 quanvolution filter."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter224Quantum()
        # For 224×224 with patch_size=3, stride=2: 111×111 patches
        self.linear = nn.Linear(4 * 111 * 111, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumResidualBlock", "QuanvolutionFilter224Quantum", "QuanvolutionClassifierQuantum"]
