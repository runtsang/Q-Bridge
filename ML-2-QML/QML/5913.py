"""Quantum counterpart that mirrors the classical hybrid model but replaces the attention
block with a variational quantum self‑attention circuit.  The quantum module is
designed to be drop‑in compatible with the classical one for comparative studies.

Key design choices:
- A 4‑qubit encoder maps the pooled classical features into a quantum state.
- A variational circuit (RandomLayer + trainable RX/RY/RZ/CRX) implements the
  *quantum* self‑attention mechanism.
- Measurement in the Pauli‑Z basis produces a 4‑dimensional vector that is
  post‑processed by a classical linear head.

The module follows the *combination* scaling paradigm by coupling a quantum
sub‑network with a lightweight classical head, enabling efficient hybrid
training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# -- Quantum self‑attention layer ---------------------------------------------
class QuantumSelfAttention(tq.QuantumModule):
    """
    Variational circuit that mimics a self‑attention operation.
    Parameters are trainable RY, RZ, RX, and CRX gates applied to 4 qubits.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Random layer for expressivity
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        # Trainable single‑qubit rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Trainable two‑qubit entangling gate
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Random feature map
        self.random_layer(qdev)
        # Parameterised rotations
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        # Additional entanglement
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])

# -- Full hybrid quantum model -----------------------------------------------
class HybridQuantumNatAttention(tq.QuantumModule):
    """
    Quantum‑inspired hybrid model that mirrors the classical `HybridQuantumNatAttention`.
    The encoder maps pooled classical features into the quantum device,
    the QuantumSelfAttention block processes them, and the output is passed
    through a classical linear head.
    """

    def __init__(self, embed_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.n_wires = 4
        # Classical encoder: same as reference (4x4_ryzxy)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        # Quantum self‑attention block
        self.q_attn = QuantumSelfAttention()
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Sequential(
            nn.Linear(self.n_wires, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Average‑pool input to a 16‑dim vector.
          2. Encode into 4 qubits.
          3. Apply quantum self‑attention circuit.
          4. Measure to obtain a 4‑dim vector.
          5. Classical head maps to final logits.
        """
        bsz = x.shape[0]
        # Classical pooling
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Encode classical features
        self.encoder(qdev, pooled)
        # Variational attention block
        self.q_attn(qdev)
        # Measurement
        out = self.measure(qdev)
        # Classical head
        out = self.head(out)
        return self.norm(out)


__all__ = ["HybridQuantumNatAttention"]
