import torch
import torch.nn as nn
import torchquantum as tq

class QuanvolutionRegressionModel(tq.QuantumModule):
    """
    Quantum quanvolution filter with a variational readout followed by an MLP regression head.
    """
    def __init__(self,
                 n_wires: int = 4,
                 conv_out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 hidden_dim: int = 64,
                 out_features: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.conv_out_channels = conv_out_channels

        # Encoder: map 4‑pixel patches to qubit states
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        feature_map_size = 28 // stride  # 14
        self.feature_dim = conv_out_channels * feature_map_size * feature_map_size
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        features = torch.cat(patches, dim=1)
        return self.mlp(features).squeeze(-1)

__all__ = ["QuanvolutionRegressionModel"]
