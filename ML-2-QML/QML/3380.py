import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F
import numpy as np

class HybridSelfAttention(tq.QuantumModule):
    """Hybrid CNN + variational self‑attention implemented on a 4‑qubit device."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        
        # Classical encoder (similar to QFCModel)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    class QLayer(tq.QuantumModule):
        """Quantum layer embedding a variational self‑attention circuit."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Rotation gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            # Entanglement gates (CRX) – emulate the classical CRX entanglement
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Random feature embedding
            self.random_layer(qdev)
            # Apply rotation gates on each wire
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=2)
            # Entangle adjacent wires with CRX
            self.crx0(qdev, wires=[0, 1])
            self.crx0(qdev, wires=[1, 2])
            self.crx0(qdev, wires=[2, 3])
            # Optional additional gates from the original QFCModel
            tqf.hadamard(qdev, wires=3)
            tqf.sx(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical pooling to match encoder input dimension
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Encode classical features into quantum state
        self.encoder(qdev, pooled)
        # Quantum self‑attention layer
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridSelfAttention"]
