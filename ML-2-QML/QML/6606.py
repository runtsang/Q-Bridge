import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class VariationalLayer(tq.QuantumModule):
    """Depth‑2 variational ansatz that mirrors the quantum circuit from the reference.

    The layer first applies a random unitary (to provide a data‑dependent basis), then
    performs an RX encoding, followed by two rounds of RY rotations and CZ entangling
    gates.  The structure is intentionally simple yet expressive enough to capture
    non‑linear interactions between the four qubits.
    """
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry_layers = [tq.RY(has_params=True, trainable=True) for _ in range(depth)]
        self.cz = tq.CZ()  # entangling gate, no parameters

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)  # simple encoding on qubit 0
        for ry in self.ry_layers:
            for wire in range(self.n_wires):
                ry(qdev, wires=wire)
            # CZ entanglement between adjacent qubits
            for wire in range(self.n_wires - 1):
                self.cz(qdev, wires=[wire, wire + 1])

class QFCModel(tq.QuantumModule):
    """Quantum‑NAT hybrid that encodes image patches into a 4‑qubit register,
    applies a variational circuit, measures Pauli‑Z observables, and maps the
    resulting expectation values to 2‑class logits via a classical linear head.
    The architecture preserves the 4‑qubit interface of the original Quantum‑NAT
    while adding a classification head that can be trained jointly with the
    quantum parameters.
    """
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = VariationalLayer(n_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)
        self.classifier = nn.Linear(n_wires, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2‑class logits derived from the measurement statistics."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pool the image to a 16‑dim vector and feed it into the encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        logits = self.classifier(out)
        return logits

__all__ = ["QFCModel", "VariationalLayer"]
