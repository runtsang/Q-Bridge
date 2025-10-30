import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum variant of the enhanced Quantum‑NAT model.

    Features:
    * 4‑wire Ry/Z/X/Y encoder.
    * Variational layer with adaptive depth controlled by a schedule tensor.
    * Measurement of all qubits in the Pauli‑Z basis.
    """

    class VariationalLayer(tq.QuantumModule):
        def __init__(self, max_depth: int, n_wires: int = 4):
            super().__init__()
            self.max_depth = max_depth
            self.n_wires = n_wires
            # Parameterised single‑qubit rotations for each depth layer
            self.rotations = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires, 3)) for _ in range(max_depth)]
            )

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, depth: int):
            for idx in range(depth):
                params = self.rotations[idx]
                for wire in range(self.n_wires):
                    tqf.rx(qdev, params[wire, 0], wires=wire, static=self.static_mode, parent_graph=self.graph)
                    tqf.ry(qdev, params[wire, 1], wires=wire, static=self.static_mode, parent_graph=self.graph)
                    tqf.rz(qdev, params[wire, 2], wires=wire, static=self.static_mode, parent_graph=self.graph)
                # Entangle adjacent qubits
                for w in range(0, self.n_wires - 1, 2):
                    tq.CNOT(qdev, wires=[w, w+1], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, max_depth: int = 10, depth_schedule: torch.Tensor | None = None):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.VariationalLayer(max_depth=max_depth, n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.out_bn = nn.BatchNorm1d(self.n_wires)
        self.depth_schedule = depth_schedule

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pre‑processing: global average pooling to 16 features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Determine depth from schedule or use maximum
        depth = int(self.depth_schedule.item()) if self.depth_schedule is not None else self.q_layer.max_depth
        self.q_layer(qdev, depth=depth)
        out = self.measure(qdev)
        return self.out_bn(out)

__all__ = ["QuantumNATEnhanced"]
