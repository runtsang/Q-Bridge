"""Quantum variant of the QFCModelExtended. It encodes a 4‑qubit circuit with a
variational layer whose depth can be tuned. The input image is first
average‑pooled to a 16‑dimensional vector and then encoded into the qubits
via a fixed Ry‑Rz‑X‑Y pattern. The circuit is followed by a depth‑controlled
variational block that learns expressive transformations. Finally,
all qubits are measured in the Pauli‑Z basis and batch‑normalised.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModelExtended(tq.QuantumModule):
    """Quantum model that mirrors the classical QFCModelExtended but replaces
    the fully‑connected head with a variational quantum circuit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            # Parameterised single‑qubit gates
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Two‑qubit entangling gate
            self.cx = tq.CX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Randomised initialisation
            self.random_layer(qdev)
            # Apply a depth‑controlled variational block
            for d in range(self.depth):
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
                # Entangle neighbouring qubits in a ring
                for w in range(self.n_wires):
                    self.cx(qdev, wires=[w, (w + 1) % self.n_wires])
            # Optional single‑qubit rotations
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)

    def __init__(self,
                 n_wires: int = 4,
                 depth: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer(n_wires=self.n_wires, depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Average‑pool the image to 16 features per sample
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModelExtended"]
