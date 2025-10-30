import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionNet(tq.QuantumModule):
    """
    Quantum hybrid network that applies a variational quantum circuit
    to 2×2 patches of a 3‑channel image. The circuit is fully
    trainable and uses a parameter‑tuned entangling layer.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.n_wires = in_channels * 4  # 3 channels × 4 pixels per patch
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )
        self.var_layer = tq.ParameterizedCircuit(
            params=torch.nn.Parameter(torch.randn(self.n_wires * 2)),
            circuit=lambda dev, params: tq.Circuit(
                [tq.RY(params[i], wires=i) for i in range(self.n_wires)] +
                [tq.CNOT(i, (i + 1) % self.n_wires) for i in range(self.n_wires)] +
                [tq.RY(params[self.n_wires + i], wires=i) for i in range(self.n_wires)]
            )
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.fc = nn.Linear(self.n_wires * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 3, 28, 28)
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, :, r:r+2, c:c+2]  # shape (bsz, 3, 2, 2)
                patch = patch.reshape(bsz, -1)  # shape (bsz, 12)
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.var_layer(qdev, self.var_layer.params)
                measurement = self.measure(qdev)
                patches.append(measurement)
        features = torch.cat(patches, dim=1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
