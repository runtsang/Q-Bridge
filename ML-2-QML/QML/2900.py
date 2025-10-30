import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATModel(tq.QuantumModule):
    """
    Quantum‑hybrid model that mirrors the classical HybridNATModel while
    replacing the filter and the final feature extractor with quantum
    circuits.  The encoder maps the pooled image into a 4‑qubit state,
    a parameterised QLayer acts as a feature extractor, and a
    lightweight QuantumFilter contributes an additional scalar
    feature that is broadcast‑added to the main measurement.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    class QuantumFilter(tq.QuantumModule):
        """
        Minimal quantum filter that produces a scalar feature by measuring
        all qubits after a random unitary.  The result is averaged over
        qubits and broadcast‑added to the main output.
        """
        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliX)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            out = self.measure(qdev)
            return out.mean(dim=1, keepdim=True)  # (bsz, 1)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.filter = self.QuantumFilter()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Main quantum device for feature extraction
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        main_out = self.measure(qdev)  # (bsz, n_wires)

        # Separate device for the lightweight filter
        qdev_filter = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        filter_out = self.filter(qdev_filter)  # (bsz, 1)

        # Broadcast‑add filter scalar to all qubits
        out = main_out + filter_out

        return self.norm(out)

__all__ = ["HybridNATModel"]
