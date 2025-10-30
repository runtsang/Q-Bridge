import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATModel(tq.QuantumModule):
    """Quantum‑enhanced NAT model with a parametric variational circuit."""
    class VariationalBlock(tq.QuantumModule):
        """Parameterised variational layer with entangling gates."""
        def __init__(self, n_wires: int, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                layer = nn.ModuleDict({
                    "rx": tq.RX(has_params=True, trainable=True),
                    "ry": tq.RY(has_params=True, trainable=True),
                    "rz": tq.RZ(has_params=True, trainable=True),
                    "cnot": tq.CNOT()
                })
                self.layers.append(layer)
        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in self.layers:
                for gate in ("rx", "ry", "rz"):
                    layer[gate](qdev)
                for i in range(self.n_wires):
                    layer["cnot"](qdev, wires=[i, (i + 1) % self.n_wires])
    def __init__(self, n_wires: int = 4, measurement: str = "Z"):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.var_block = self.VariationalBlock(n_wires)
        if measurement == "Z":
            self.measure = tq.MeasureAll(tq.PauliZ)
        elif measurement == "X":
            self.measure = tq.MeasureAll(tq.PauliX)
        else:
            raise ValueError("Unsupported measurement basis")
        self.norm = nn.BatchNorm1d(n_wires)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.var_block(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridNATModel"]
