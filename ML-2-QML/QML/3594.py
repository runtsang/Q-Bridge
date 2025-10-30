"""Hybrid quantum kernel built upon a QCNN‑style ansatz.

The :class:`HybridKernelModel` mirrors the classical implementation by
encoding data into an 8‑qubit device with a QCNN circuit, then computing the
absolute overlap of the resulting state vectors.  Parameters are derived
directly from the input samples, providing a data‑driven quantum kernel
compatible with TorchQuantum.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QCNNAnsatz(tq.QuantumModule):
    """Quantum analogue of the classical QCNN feature extractor.

    The ansatz applies a sequence of two‑qubit convolution and pooling blocks
    to an 8‑qubit device.  Data are encoded via Ry rotations and the same
    three parameters (first three features) are reused for every block,
    keeping the circuit compact while retaining sensitivity to the input.
    """
    def __init__(self, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.blocks = self._build_block_list()

    @staticmethod
    def _conv_block(q_device: tq.QuantumDevice, wires: List[int], params: torch.Tensor) -> None:
        func_name_dict["rz"](q_device, wires=[wires[1]], params=params[:,0])
        func_name_dict["cx"](q_device, wires=wires)
        func_name_dict["rz"](q_device, wires=[wires[0]], params=params[:,1])
        func_name_dict["ry"](q_device, wires=[wires[1]], params=params[:,2])
        func_name_dict["cx"](q_device, wires=wires)

    @staticmethod
    def _pool_block(q_device: tq.QuantumDevice, wires: List[int], params: torch.Tensor) -> None:
        func_name_dict["rz"](q_device, wires=[wires[1]], params=params[:,0])
        func_name_dict["cx"](q_device, wires=wires)
        func_name_dict["rz"](q_device, wires=[wires[0]], params=params[:,1])
        func_name_dict["ry"](q_device, wires=[wires[1]], params=params[:,2])
        func_name_dict["cx"](q_device, wires=wires)

    def _build_block_list(self) -> List[Dict]:
        blocks: List[Dict] = []
        # Convolution layers (8 qubits)
        for pair in [(0,1),(2,3),(4,5),(6,7)]:
            blocks.append({"type":"conv","wires":list(pair),"n_params":3})
        # Pooling layers (8 qubits)
        for pair in [(0,1),(2,3),(4,5),(6,7)]:
            blocks.append({"type":"pool","wires":list(pair),"n_params":3})
        # Convolution layers (4 qubits)
        for pair in [(4,5),(6,7)]:
            blocks.append({"type":"conv","wires":list(pair),"n_params":3})
        # Pooling layers (4 qubits)
        for pair in [(4,5),(6,7)]:
            blocks.append({"type":"pool","wires":list(pair),"n_params":3})
        # Convolution layer (2 qubits)
        blocks.append({"type":"conv","wires":[6,7],"n_params":3})
        # Pooling layer (2 qubits)
        blocks.append({"type":"pool","wires":[6,7],"n_params":3})
        return blocks

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode two samples and compute the overlap of the resulting state."""
        self.q_device.reset_states(x.shape[0])

        # Encode x
        for i in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=[i], params=x[:, i])

        # Apply blocks with parameters from x (reuse first three features)
        for block in self.blocks:
            params = x[:, :3]
            if block["type"] == "conv":
                self._conv_block(self.q_device, block["wires"], params)
            else:
                self._pool_block(self.q_device, block["wires"], params)

        # Encode y with negative parameters
        for i in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=[i], params=-y[:, i])

        # Apply blocks in reverse with parameters from y (reuse first three features)
        for block in reversed(self.blocks):
            params = -y[:, :3]
            if block["type"] == "conv":
                self._conv_block(self.q_device, block["wires"], params)
            else:
                self._pool_block(self.q_device, block["wires"], params)

class HybridKernelModel(tq.QuantumModule):
    """Quantum kernel that mirrors the classical :class:`HybridKernelModel`."""

    def __init__(self, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = QCNNAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(x, y)
        return torch.abs(self.ansatz.q_device.states.view(-1)[0])

def kernel_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Gram matrix for two collections of raw samples.

    Parameters
    ----------
    a, b : np.ndarray
        2‑D arrays of shape (n_samples, n_features).
    """
    model = HybridKernelModel()
    a_t = torch.from_numpy(a.astype(np.float32))
    b_t = torch.from_numpy(b.astype(np.float32))
    return np.array([[model(x, y).item() for y in b_t] for x in a_t])

__all__ = ["QCNNAnsatz", "HybridKernelModel", "kernel_matrix"]
