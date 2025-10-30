"""Quantum kernel and variational layer for hybrid training."""
from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Quantum Kernel --------------------- #
class QuantumRBFKernel(tq.QuantumModule):
    """
    Implements a quantum kernel by encoding two inputs with a shared ansatz
    and measuring the overlap of the resulting states.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple parameter‑free encoding: Ry gates
        self.ansatz = tq.QuantumModule()
        self.ansatz.add(tq.RY(has_params=True, trainable=False))
        self.ansatz.add(tq.RY(has_params=True, trainable=False))
        self.ansatz.add(tq.RY(has_params=True, trainable=False))
        self.ansatz.add(tq.RY(has_params=True, trainable=False))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        qdev.reset_states(x.shape[0])
        # Encode x
        for i, gate in enumerate(self.ansatz):
            gate(qdev, wires=[i], params=x[:, i] if gate.num_params else None)
        # Reverse encode y with negative parameters
        for i, gate in reversed(list(enumerate(self.ansatz))):
            gate(qdev, wires=[i], params=-y[:, i] if gate.num_params else None)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumRBFKernel()
    return np.array([[kernel.kernel_value(x, y).item() for y in b] for x in a])

# --------------------- Quantum Fully Connected Layer --------------------- #
class QuantumFCLayer(tq.QuantumModule):
    """Variational layer inspired by Quantum‑NAT."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tq.f.hadamard(qdev, wires=3)
        tq.f.sx(qdev, wires=2)
        tq.f.cnot(qdev, wires=[3, 0])

class QuantumFCModel(tq.QuantumModule):
    """
    End‑to‑end quantum model that mirrors the classical CNN architecture
    but operates on pooled image features encoded into qubits.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QuantumFCLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumRBFKernel", "quantum_kernel_matrix", "QuantumFCLayer", "QuantumFCModel"]
