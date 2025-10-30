"""Quantum hybrid model combining the Quantum‑NAT encoder with a QCNN‑style ansatz."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torchquantum as tq
from torchquantum import encoder_op_list_name_dict
import torchquantum.functional as tqf


class HybridQuantumNet(tq.QuantumModule):
    """
    Quantum hybrid network that fuses the Quantum‑NAT encoder with a QCNN‑style variational ansatz.

    Architecture
    ------------
    * **Encoder** – Uses a 4‑qubit ``GeneralEncoder`` (4x4_ryzxy) to map the classical feature vector
      into the quantum state.
    * **Ansatz** – A hierarchy of convolutional and pooling layers (QCNN) built with
      parametric gates (RX, RY, RZ, CRX) and CNOTs.  The circuit depth mirrors
      the classical QCNNModel (three conv/pool stages).
    * **Measurement** – All‑qubit Pauli‑Z measurement followed by a BatchNorm for post‑processing.
    """

    class _QLayer(tq.QuantumModule):
        """Internal variational layer that implements a single QCNN convolution + pool block."""

        def __init__(self, num_qubits: int):
            super().__init__()
            self.num_qubits = num_qubits

            # Random feature layer to initialise the parameters
            self.random_layer = tq.RandomLayer(
                n_ops=20, wires=list(range(num_qubits))
            )

            # Parametric gates per qubit
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            """Apply a QCNN‑style convolution followed by a pooling operation."""
            self.random_layer(qdev)

            # Convolution block (three parametric rotations per pair)
            for q in range(0, self.num_qubits, 2):
                self.rz(qdev, wires=q)
                self.ry(qdev, wires=q + 1)
                self.crx(qdev, wires=[q, q + 1])

            # Pooling block – collapse pairs with a CNOT and a single rotation
            for q in range(0, self.num_qubits, 2):
                tqf.cnot(qdev, wires=[q, q + 1])
                self.rz(qdev, wires=q)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self._QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, features)`` where ``features`` is 8
            (matching the 4‑qubit feature map of size 2ⁿ).
        """
        bsz = x.shape[0]
        # Prepare a quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )

        # Classical pooling to match the 4‑qubit feature vector
        pooled = F.avg_pool1d(x.unsqueeze(1), kernel_size=2).view(bsz, self.n_wires)

        # Encode classical data into the quantum state
        self.encoder(qdev, pooled)

        # Variational ansatz
        self.q_layer(qdev)

        # Measurement
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridQuantumNet"]
