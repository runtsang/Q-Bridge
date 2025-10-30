import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Depth‑controlled, parameter‑efficient quantum layer with an entanglement block."""

    def __init__(self, n_wires: int, depth: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth

        # Random layer scaled with depth
        self.random_layer = tq.RandomLayer(
            n_ops=50 * self.depth,
            wires=list(range(self.n_wires))
        )

        # Small set of trainable gates
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

        # Entanglement block: two CNOTs that connect non‑adjacent wires
        self.entanglement1 = tq.CNOT(wires=[0, 1])
        self.entanglement2 = tq.CNOT(wires=[2, 3])

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)

        # Parameterised single‑qubit rotations
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])

        # Additional fixed operations (Hadamard, SX, CNOT)
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

        # Entanglement block
        self.entanglement1(qdev)
        self.entanglement2(qdev)


class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum variant of the enhanced NAT model with a learnable encoder and depth‑controlled circuit."""

    def __init__(self, trainable_features: bool = True, depth: int = 3):
        super().__init__()
        self.trainable_features = trainable_features
        self.n_wires = 4

        # Encoder that maps a 16‑dim classical vector to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Parameter‑efficient quantum layer
        self.q_layer = QLayer(self.n_wires, depth=depth)

        # Measurement of all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Freeze or unfreeze encoder and quantum layer
        self.toggle_features(self.trainable_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )

        # Classical preprocessing: average pooling and flattening
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Quantum processing
        self.q_layer(qdev)

        # Output
        out = self.measure(qdev)
        return self.norm(out)

    def toggle_features(self, flag: bool):
        """Enable or disable gradients for the encoder and quantum layer."""
        self.trainable_features = flag
        for p in self.encoder.parameters():
            p.requires_grad = flag
        for p in self.q_layer.parameters():
            p.requires_grad = flag


__all__ = ["QuantumNATEnhanced"]
