import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class QuantumNATPlus(tq.QuantumModule):
    """
    Quantum‑augmented version of the extended classical model.
    It keeps the same input shape and 4‑output interface but replaces the FC head
    with a parameter‑efficient QAOA‑style Ansatz that is trained jointly with the
    preceding classical encoder.
    """
    class QAOALayer(tq.QuantumModule):
        """
        QAOA‑style variational layer with 4 qubits.
        Uses a single layer of alternating X‑rotations and ZZ interactions.
        """
        def __init__(self, n_wires: int = 4, n_layers: int = 2):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT
            self.cx = tq.CX

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                # parameterized single‑qubit rotations
                for w in range(self.n_wires):
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
                # entangling layer
                for w in range(self.n_wires - 1):
                    self.cnot(qdev, wires=[w, w + 1])
                # wrap‑around for 4 qubits
                self.cnot(qdev, wires=[self.n_wires - 1, 0])

    def __init__(self, in_channels: int = 1, out_features: int = 4, use_residual: bool = True):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QAOALayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(out_features)

        # classical depth‑wise separable conv as in the classical version
        self.dw_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, groups=in_channels)
        self.dw_bn = nn.BatchNorm2d(8)
        self.dw_relu = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(8, 8, kernel_size=1)
        self.pw_bn = nn.BatchNorm2d(8)
        self.pw_relu = nn.ReLU(inplace=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res_conv = nn.Conv2d(in_channels, 16, kernel_size=1) if use_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # classical feature extraction
        dw = self.dw_conv(x)
        dw = self.dw_bn(dw)
        dw = self.dw_relu(dw)
        pw = self.pw_conv(dw)
        pw = self.pw_bn(pw)
        pw = self.pw_relu(pw)
        if self.res_conv is not None:
            pw = pw + self.res_conv(x)
        pw = self.conv_block(pw)
        flat = pw.view(pw.size(0), -1)
        # encode flattened features into qubits
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device, record_op=True)
        # average pooling to match 4 qubits
        pooled = F.avg_pool2d(flat.view(bsz, 4, -1), kernel_size=1).view(bsz, 4)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATPlus"]
