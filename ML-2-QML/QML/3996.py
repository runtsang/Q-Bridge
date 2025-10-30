import torch
import torch.nn as nn
import torchquantum as tq

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum analogue of the 2×2 patch encoder."""
    def __init__(self, n_wires: int = 4):
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
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 4] – one patch
        self.encoder(qdev, x)
        self.q_layer(qdev)
        return self.measure(qdev)

class QuantumRegressionHead(tq.QuantumModule):
    """Quantum regression head from the second reference, adapted to 4 wires."""
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def forward(self, qdev: tq.QuantumDevice, state_batch: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

class QuanvolutionRegression(tq.QuantumModule):
    """Full quantum hybrid: quanvolution filter + quantum regression head."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.feature_extractor = QuantumQuanvolutionFilter(n_wires=n_wires)
        self.regressor = QuantumRegressionHead(num_wires=n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 1, 28, 28]
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.feature_extractor.n_wires,
                                bsz=bsz, device=x.device)

        # Apply the patch‑wise quanvolution filter
        measurements = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev.reset()
                meas = self.feature_extractor(qdev, patch)
                measurements.append(meas)

        # Concatenate all patch measurements: shape [bsz, 4*14*14]
        patch_features = torch.cat(measurements, dim=1)

        # Pool across patches to obtain a 4‑dimensional vector per sample
        patch_features = patch_features.view(bsz, 14 * 14, 4).mean(dim=1)

        # Run the quantum regression head on the pooled features
        return self.regressor(qdev, patch_features)

__all__ = ["QuanvolutionRegression"]
