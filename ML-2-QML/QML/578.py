"""Quantum‑NAT model with variational circuit and SWAP‑based encoding.

The model extends the seed by adding a tunable entanglement depth
to the variational block and a SWAP‑based data encoding that
improves expressivity on small quantum devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class SWAPEncoder(tq.QuantumModule):
    """SWAP‑based data encoding.

    For each qubit, a data‑dependent RX rotation is applied,
    followed by a sequence of SWAP operations that entangle
    neighbouring qubits.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rx = tq.RX(has_params=True, trainable=False)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor):
        # data shape: (bsz, n_wires)
        for i in range(self.n_wires):
            self.rx(qdev, data[:, i], wires=i)
        # SWAP between consecutive qubits
        for i in range(self.n_wires):
            tqf.cnot(qdev, wires=[i, (i + 1) % self.n_wires], parent_graph=self.graph)
            tqf.cnot(qdev, wires=[(i + 1) % self.n_wires, i], parent_graph=self.graph)
            tqf.cnot(qdev, wires=[i, (i + 1) % self.n_wires], parent_graph=self.graph)

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum fully‑connected model with SWAP encoding and configurable entanglement depth.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits (default 4).
    ent_depth : int, optional
        Depth of entangling layers in the variational block (default 2).
    """
    class VariationalBlock(tq.QuantumModule):
        def __init__(self, n_wires: int, ent_depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.ent_depth = ent_depth
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for i in range(self.n_wires):
                self.rx(qdev, wires=i)
                self.ry(qdev, wires=i)
                self.rz(qdev, wires=i)
            # Entanglement
            for d in range(self.ent_depth):
                for i in range(self.n_wires):
                    tqf.cnot(qdev, wires=[i, (i + 1) % self.n_wires], parent_graph=self.graph)

    def __init__(self, n_wires: int = 4, ent_depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.ent_depth = ent_depth
        self.encoder = SWAPEncoder(n_wires)
        self.var_block = self.VariationalBlock(n_wires, ent_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Pool input to match qubit count
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 4, 4)
        data = pooled.mean(dim=2)  # shape (bsz, 4)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, data)
        self.var_block(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

__all__ = ["QuantumNATEnhanced"]
