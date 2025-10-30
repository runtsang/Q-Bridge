"""Quantum autoencoder module inspired by the QML reference and Autoencoder.

The module encodes classical inputs into a quantum state, applies a variational
layer, and measures a reduced set of qubits to produce a latent representation.
"""

import torch
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn as nn

class QFCModel(tq.QuantumModule):
    """Quantum-only model that mirrors the classical QFCModel structure.

    It accepts a classical tensor, encodes it into a quantum circuit,
    applies a random variational layer, and measures all qubits to
    produce a latent vector of size `n_wires`.
    """

    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical input `x` into a quantum state and return the
        measured latent vector.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModel"]
