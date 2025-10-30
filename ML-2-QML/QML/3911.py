import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumNAT(tq.QuantumModule):
    """Quantum counterpart of the classical HybridQuantumNAT.

    It retains the same CNN backbone but replaces the final
    fully‑connected projection with a parameterised quantum circuit.
    The quantum layer is inspired by the QFCModel in the first seed
    and the FCL example in the second seed, providing a genuine
    quantum contribution to the output."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_qubits=n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_qubits)

    class QLayer(tq.QuantumModule):
        """Parameterised quantum circuit that acts on the encoded
        classical features.  It mirrors the structure of the
        original QFCModel's QLayer but uses a loop for clarity."""
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_qubits)))
            # 4 trainable rotation gates
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
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz,
                                device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """A minimal quantum analog of a fully‑connected layer.

    The circuit applies a Ry rotation to each qubit and measures
    Pauli‑Z expectation values.  It can be inserted into the
    main model as a drop‑in replacement for a classical linear
    projection."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for i in range(self.n_qubits):
            tq.RY(has_params=True, trainable=True)(qdev, wires=i)
        return self.measure(qdev)
__all__ = ["HybridQuantumNAT", "QuantumFullyConnectedLayer"]
