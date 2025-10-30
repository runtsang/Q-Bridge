"""Hybrid quantum model with data encoding, variational circuit and quantum kernel.

The class exposes a forward pass that maps classical data onto a small
quantum register, applies a depth‑controlled variational ansatz, and
measures in the computational basis.  Additionally, a quantum‑kernel
method is provided for kernel‑based inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence


class KernalAnsatz(tq.QuantumModule):
    """Quantum feature‑map that applies a list of gates fed by two
    classical vectors (x and y)."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QFCModel(tq.QuantumModule):
    """Quantum model that encodes data, runs a variational circuit and
    measures all qubits.  It also offers a quantum kernel evaluation.
    """

    class QLayer(tq.QuantumModule):
        """Depth‑controlled variational block with random, RY and CZ gates."""

        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            # One RY per qubit per depth
            self.ry_layers = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(depth * n_wires)]
            )
            # CZ gates between neighboring qubits for each depth
            self.cz_layers = nn.ModuleList(
                [tq.CZ() for _ in range(depth * (n_wires - 1))]
            )

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            idx = 0
            for d in range(self.depth):
                # RY on each qubit
                for w in range(self.n_wires):
                    self.ry_layers[idx](qdev, wires=w)
                    idx += 1
                # CZ between neighboring qubits
                for w in range(self.n_wires - 1):
                    self.cz_layers[d * (self.n_wires - 1) + w](qdev, wires=[w, w + 1])

    def __init__(self, n_qubits: int = 4, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        # Classical encoding using a small fixed ansatz
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Variational block
        self.q_layer = self.QLayer(n_qubits, depth)

        # Measurement and output normalization
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_qubits)

        # Quantum kernel ansatz
        self.kernel_ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, variational update, measurement."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device, record_op=True)
        # Simple pooling to a 16‑dimensional feature vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum kernel between two input vectors."""
        # Reshape inputs to 1‑D feature vectors
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits)
        self.kernel_ansatz(qdev, x, y)
        return torch.abs(qdev.states.view(-1)[0])
