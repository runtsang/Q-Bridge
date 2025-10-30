import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class VariationalBlock(tq.QuantumModule):
    """Layer‑wise variational circuit with parameterised rotations and CNOT entanglement."""
    def __init__(self, n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        # Three rotation gates per qubit
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
            self.rz(qdev, wires=w)
        # Entangle qubits in a linear chain
        for i in range(self.n_wires - 1):
            self.cnot(qdev, wires=[i, i + 1])

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum version of the enhanced NAT model.
    Implements a trainable measurement weight layer and a variational circuit.
    Returns a tuple (logits, aux_logits) mirroring the classical counterpart.
    """
    def __init__(self, n_wires=4, num_classes=4, depth=3):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Stack multiple variational blocks
        self.vqc = nn.ModuleList([VariationalBlock(n_wires) for _ in range(depth)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Trainable weight layer mapping measurement outcomes to logits
        self.weight_layer = nn.Linear(n_wires, num_classes, bias=True)
        self.norm = nn.BatchNorm1d(num_classes)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        # Prepare device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode classical data
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Variational layers
        for layer in self.vqc:
            layer(qdev)

        # Measurement
        out = self.measure(qdev)  # shape (bsz, n_wires)

        # Auxiliary logits (raw measurement)
        aux_logits = out

        # Trainable weighted logits
        logits = self.weight_layer(out)
        return self.norm(logits), self.norm(aux_logits)

__all__ = ["QuantumNATEnhanced"]
