"""
HybridNATModel: Quantum module implementing data encoding, a depth‑controlled variational ansatz,
and measurement, with optional quantum kernel evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import QuantumDevice
from torchquantum.operators import PauliZ
from torchquantum.modules import QuantumModule


class HybridNATModel(QuantumModule):
    """
    Quantum counterpart to the classical HybridNATModel.
    Features:
    * General encoder (e.g., 4×4_ryzxy) for data embedding.
    * A variational ansatz composed of Y rotations and CNOTs, depth‑controlled.
    * Measurement of all qubits in the Pauli‑Z basis.
    * Batch‑norm on the output to match the classical model.
    """

    def __init__(self, n_wires: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Data encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Variational ansatz
        self.ansatz = tq.Sequential(
            *[tq.RY(has_params=True, trainable=True) for _ in range(depth)],
            *[tq.CNOT() for _ in range(n_wires - 1)],
        )
        # Measurement
        self.measure = tq.MeasureAll(PauliZ)
        # Output normalisation
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode, variational layer, measurement.
        """
        bsz = x.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Reduce image to a vector compatible with the encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.ansatz(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a quantum kernel between two batches of inputs.
        This implementation uses state overlap on the encoded and ansatzed states.
        """
        # Concatenate inputs for batch processing
        combined = torch.cat([x, y], dim=0)
        bsz = combined.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(combined, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.ansatz(qdev)
        # Compute inner‑product kernel via states
        states = qdev.states.clone()
        # Separate states for x and y
        states_x = states[: x.shape[0]]
        states_y = states[x.shape[0] :]
        # Overlap matrix: |<x|y>|^2
        overlaps = torch.abs(torch.mm(states_x, states_y.conj().t())).diag()
        return overlaps

__all__ = ["HybridNATModel"]
