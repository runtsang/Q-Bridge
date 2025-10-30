"""QuantumNAT_Advanced: Quantum module with variational circuits and multi-task heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNAT_Advanced(tq.QuantumModule):
    """A modernized quantum module that extends the original Quantum‑NAT architecture."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.param_gates = nn.ModuleList([
                tq.RX(has_params=True, trainable=True),
                tq.RY(has_params=True, trainable=True),
                tq.RZ(has_params=True, trainable=True),
                tq.CRX(has_params=True, trainable=True)
            ])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            # Apply parametric gates on each wire
            for i, gate in enumerate(self.param_gates):
                gate(qdev, wires=i % self.n_wires)
            # Additional fixed gates
            tqf.hadamard(qdev, wires=0)
            tqf.cnot(qdev, wires=[0, 1])

    def __init__(self, num_tasks: int = 4, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.task_heads = nn.ModuleDict({
            f"task_{i}": nn.Linear(self.n_wires, 4) for i in range(num_tasks)
        })

    def forward(self, x: torch.Tensor) -> dict:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pooling of image features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        outputs = {name: head(out) for name, head in self.task_heads.items()}
        return outputs

__all__ = ["QuantumNAT_Advanced"]
