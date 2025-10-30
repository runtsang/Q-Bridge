import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from typing import Sequence
import torch.nn as nn

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that encodes data via a GeneralEncoder and a RandomLayer."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50, encoder_name: str = "4x4_ryzxy") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        self.encoder(q_device, x)
        self.random_layer(q_device)

class Kernel(tq.QuantumModule):
    """Wrapper around KernalAnsatz to provide a callable quantum kernel."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50, encoder_name: str = "4x4_ryzxy") -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(n_wires, n_ops, encoder_name)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = tq.QuantumDevice(n_wires=self.ansatz.n_wires, bsz=1, device=x.device)
        self.ansatz(device, x)
        psi_x = device.states
        device.reset_states(1)
        self.ansatz(device, y)
        psi_y = device.states
        overlap = torch.abs(torch.sum(psi_x.conj() * psi_y))**2
        return overlap

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, n_ops: int = 50, encoder_name: str = "4x4_ryzxy") -> np.ndarray:
    kernel = Kernel(n_wires, n_ops, encoder_name)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridKernelMethod(tq.QuantumModule):
    """Hybrid quantum kernel that uses a GeneralEncoder and RandomLayer, then computes overlap."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50, encoder_name: str = "4x4_ryzxy") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=x.device)
        self.encoder(device, x)
        self.random_layer(device)
        psi_x = device.states
        device.reset_states(1)
        self.encoder(device, y)
        self.random_layer(device)
        psi_y = device.states
        overlap = torch.abs(torch.sum(psi_x.conj() * psi_y))**2
        return overlap

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "HybridKernelMethod", "kernel_matrix"]
