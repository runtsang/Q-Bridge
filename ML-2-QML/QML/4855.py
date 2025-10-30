import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumBlock(tq.QuantumModule):
    """Quantum core that encodes a 16‑dim vector into 4 qubits and applies a variational layer."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.layer = self._QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = tq.nn.BatchNorm1d(self.n_wires)

    class _QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cz = tq.CZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            self.cz(qdev, wires=[1, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, x)
        self.layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridQuantumBlock"]
