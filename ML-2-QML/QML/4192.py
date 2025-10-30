import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATModel(tq.QuantumModule):
    """Quantum model that merges a quantum convolutional filter, encoder, variational layer and a quantum kernel."""
    class QConvFilter(tq.QuantumModule):
        """Simple quantum filter that encodes a patch into a quantum state."""
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor):
            self.random_layer(qdev, data)

    class QKernel(tq.QuantumModule):
        """Quantum kernel based on a fixed random circuit."""
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.ansatz(self.q_device, x)
            self.ansatz(self.q_device, y, reverse=True)
            return torch.abs(self.q_device.states.view(-1)[0])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.conv_filter = self.QConvFilter()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.kernel = self.QKernel()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires + 1)

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Generate quantum convolutional filter features per patch
        patches = x.unfold(2, self.conv_filter.n_wires, 1).unfold(3, self.conv_filter.n_wires, 1)
        patches = patches.contiguous().view(bsz, -1, self.conv_filter.n_wires)
        measure_all = tq.MeasureAll(tq.PauliZ)
        patch_vals = []
        for i in range(bsz):
            qdev_patch = tq.QuantumDevice(n_wires=self.conv_filter.n_wires, bsz=patches[i].shape[0], device=x.device)
            self.conv_filter(qdev_patch, patches[i])
            vals = measure_all(qdev_patch)
            patch_vals.append(vals.mean().unsqueeze(0))
        conv_feat = torch.cat(patch_vals, dim=0).unsqueeze(-1)          # (bsz,1)
        conv_feat = conv_feat.repeat(1, self.n_wires)                    # (bsz,4)

        # Main quantum circuit
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)                                 # (bsz,4)

        # Quantum kernel similarity between conv feature and main output
        kernel_vals = torch.stack([self.kernel(conv_feat[i].unsqueeze(0), out[i].unsqueeze(0))
                                   for i in range(bsz)])
        out = torch.cat([out, kernel_vals.unsqueeze(-1)], dim=1)   # (bsz,5)
        return self.norm(out)

__all__ = ["QuantumNATModel"]
