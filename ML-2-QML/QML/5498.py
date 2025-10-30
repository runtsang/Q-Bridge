import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class ResidualBlock(tq.QuantumModule):
    """Depth‑wise residual block using classical PyTorch layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class DepthwiseSeparableCNN(tq.QuantumModule):
    """Quantum‑aware backbone that remains classical."""
    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3,
                                 stride=1, padding=1, groups=base_channels,
                                 bias=False)
        self.pw_conv = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1,
                                 bias=False)
        self.bn_pw = nn.BatchNorm2d(base_channels * 2)
        self.res_block = ResidualBlock(base_channels * 2, base_channels * 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dw_conv(out)
        out = self.pw_conv(out)
        out = self.bn_pw(out)
        out = self.relu(out)
        out = self.res_block(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

class QuantumHybridNet(tq.QuantumModule):
    """
    Quantum‑centric counterpart of the classical hybrid model.
    Mirrors the architecture but replaces the classical kernel with a variational circuit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self,
                 num_classes: int = 10,
                 regression: bool = False,
                 n_wires: int = 16,
                 device: str | torch.device | None = None) -> None:
        super().__init__()
        self.backbone = DepthwiseSeparableCNN(in_channels=1, base_channels=8)
        self.n_wires = n_wires
        # Encoder that maps a feature vector to rotations on the qubits
        encoding_ops = [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        self.encoder = tq.GeneralEncoder(encoding_ops)
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(n_wires, num_classes)
        self.regression = regression
        if regression:
            self.regressor = nn.Linear(n_wires, 1)
        self._clip_parameters()
        self.to(device or torch.device('cpu'))

    def _clip_parameters(self, bound: float = 5.0) -> None:
        """Clip linear layer weights to keep them bounded."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.clamp_(min=-bound, max=bound)
                if m.bias is not None:
                    m.bias.data.clamp_(min=-bound, max=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quantum circuit.
        Returns:
            - logits (classification) or
            - (logits, regression_output) if regression=True.
        """
        features = self.backbone(x)          # shape: (batch, 16)
        batch_size = features.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=x.device)
        self.encoder(qdev, features)
        self.q_layer(qdev)
        quantum_features = self.measure(qdev)  # shape: (batch, n_wires)
        logits = self.classifier(quantum_features)
        if self.regression:
            regress = self.regressor(quantum_features).squeeze(-1)
            return logits, regress
        return logits

__all__ = ["QuantumHybridNet"]
