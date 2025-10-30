import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum counterpart of the extended classical model."""
    class VariationalBlock(tq.QuantumModule):
        """Hardware‑efficient variational layer."""
        def __init__(self):
            super().__init__()
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cnot = tq.CNOT(has_params=False, trainable=False)
            self.cry = tq.CRY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.crx(qdev, wires=[0, 1])
            self.cnot(qdev, wires=[1, 2])
            self.cry(qdev, wires=[2, 3])
            self.crx(qdev, wires=[1, 2])
            self.cnot(qdev, wires=[0, 3])
            self.cry(qdev, wires=[3, 0])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_block = self.VariationalBlock()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.classical_head = nn.Linear(self.n_wires, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, kernel_size=6, stride=6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_block(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        out = self.classical_head(out)
        return out
