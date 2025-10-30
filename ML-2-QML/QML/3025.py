"""Quantum implementation of the hybrid architecture.

QuantumNATHybrid is a torchquantum module that encodes classical image
features into a 4‑qubit state, processes them through a variational layer
and a sampler circuit, and returns a probability distribution over four
classes.  The design integrates a general encoder, a random layer for
expressivity, and a quantum sampler based on a parameterized circuit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSampler(tq.QuantumModule):
    """Parameterized quantum sampler that produces a probability vector."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Parameterised rotation angles
        self.ry = [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        self.rz = [tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)]
        # Entangling layer
        self.cnot = tq.CNOT(wires=[0, 1], has_params=False)
        self.cnot2 = tq.CNOT(wires=[2, 3], has_params=False)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
        # Apply rotations
        for i in range(self.n_wires):
            self.ry[i](qdev, wires=i)
            self.rz[i](qdev, wires=i)
        # Entangle
        self.cnot(qdev)
        self.cnot2(qdev)

class QuantumNATHybrid(tq.QuantumModule):
    """Quantum version of the hybrid architecture."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder that maps 16‑dim pooled features to qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Variational layer inspired by the original QFCModel
        self.q_layer = self._build_variational_layer()
        # Sampler to get probability distribution
        self.sampler = QuantumSampler(self.n_wires)
        # Measurement of all qubits in computational basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def _build_variational_layer(self) -> tq.QuantumModule:
        class VariationalLayer(tq.QuantumModule):
            def __init__(self, n_wires: int = 4):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
                self.rx = [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                self.ry = [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
                self.rz = [tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)]
                self.crx = [tq.CRX(has_params=True, trainable=True) for _ in range(n_wires)]

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
                self.random_layer(qdev)
                for i in range(self.n_wires):
                    self.rx[i](qdev, wires=i)
                    self.ry[i](qdev, wires=i)
                    self.rz[i](qdev, wires=i)
                    self.crx[i](qdev, wires=[i, (i+1)%self.n_wires])

        return VariationalLayer(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pool image features to 16‑dim vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into qubits
        self.encoder(qdev, pooled)
        # Variational processing
        self.q_layer(qdev)
        # Sampler layer
        self.sampler(qdev)
        # Measure to obtain expectation values
        out = self.measure(qdev)
        # Convert to probabilities via softmax
        probs = F.softmax(out, dim=-1)
        return self.norm(probs)

__all__ = ["QuantumNATHybrid"]
