import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridNATModel(tq.QuantumModule):
    """
    Quantum counterpart of HybridNATModel.  The model encodes a 2‑D input into
    a 4‑qubit state, applies a stochastic RandomLayer followed by a small
    trainable variational block (RX/RY gates), measures all qubits, and
    projects the measurement vector to a 4‑dimensional prediction.

    Architecture mirrors the classical version so that the same training
    pipeline can be swapped between backends.
    """

    class QLayer(tq.QuantumModule):
        """Variational sub‑graph operating on all qubits."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:  # pragma: no cover
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a 16‑dimensional vector to a 4‑qubit state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4xRy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Average‑pool to 16‑dimensional vector (matching the encoder input size)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        return self.norm(out)


__all__ = ["HybridNATModel"]
