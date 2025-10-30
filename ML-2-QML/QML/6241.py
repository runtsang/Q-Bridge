"""Hybrid quantum‑classical estimator that combines a parameterized quantum circuit with
classical post‑processing.  The circuit is built from a random layer and a sequence of
trainable RX/RY/RZ/CRX gates (inspired by the Quantum‑NAT QFCModel).  After measurement
the results are fed through a small classical linear head to produce a scalar output."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridEstimatorQuantum(tq.QuantumModule):
    """Hybrid quantum‑classical estimator."""

    class QLayer(tq.QuantumModule):
        """Parameterized quantum layer consisting of a random layer and single‑qubit rotations."""
        def __init__(self, n_wires: int = 4, n_ops: int = 50):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            tqf.hadamard(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder maps classical data into quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical post‑processing head
        self.classical_head = nn.Sequential(
            nn.Linear(n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Classical input of shape ``(batch, features)``.  The first ``n_wires`` values are
            encoded into the quantum circuit; the rest are ignored.

        Returns
        -------
        torch.Tensor
            Scalar prediction per sample.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode the first n_wires features
        encoded = x[:, :self.n_wires].unsqueeze(1)
        self.encoder(qdev, encoded)
        # Apply quantum layer
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        out = self.norm(out)
        # Classical post‑processing
        return self.classical_head(out)

__all__ = ["HybridEstimatorQuantum"]
