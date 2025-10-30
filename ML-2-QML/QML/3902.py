import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumSampler(tq.QuantumModule):
    """Parameterised 2‑qubit quantum sampler producing a 2‑dim probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        # Trainable rotation angles
        self.w0 = nn.Parameter(torch.randn(1))
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        self.encoder(device, x)
        tq.RY(self.w0)(device, wires=[0])
        tq.RY(self.w1)(device, wires=[1])
        tq.CX(device, wires=[0, 1])
        tq.RY(self.w2)(device, wires=[0])
        tq.RY(self.w3)(device, wires=[1])
        measurement = self.measure(device)
        return measurement

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies trainable rotations to 2×2 image patches."""
    def __init__(self) -> None:
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
        # 8 trainable rotation angles for the 4‑qubit patch
        self.w_params = nn.Parameter(torch.randn(8))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
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
                self.encoder(device, data)
                # Apply 4 RY and 4 RZ rotations with trainable angles
                for i in range(4):
                    tq.RY(self.w_params[i])(device, wires=[i])
                for i in range(4, 8):
                    tq.RZ(self.w_params[i])(device, wires=[i-4])
                measurement = self.measure(device)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class HybridQuanvolutionSampler(tq.QuantumModule):
    """
    Hybrid quantum model that concatenates outputs from a quantum quanvolution filter
    and a quantum sampler, then classifies with a linear head.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.sampler = QuantumSampler()
        self.linear = nn.Linear(4 * 14 * 14 + 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        sampler_input = features[:, :2]
        probs = self.sampler(sampler_input)
        combined = torch.cat((features, probs), dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionSampler"]
