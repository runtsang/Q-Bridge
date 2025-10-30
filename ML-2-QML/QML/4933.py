"""Hybrid quantum convolution module using TorchQuantum, mirroring QCNN and Quantum‑NAT."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf


class HybridConv(tq.QuantumModule):
    """
    Quantum hybrid convolution model.

    The circuit consists of:
    1. A feature map encoding classical data via a 4‑parameter Ry/Z‑rotation pattern.
    2. A QCNN‑style depth‑wise convolution layer followed by a pooling layer.
    3. A Quantum‑NAT inspired fully‑connected block with a random layer and
       trainable single‑qubit gates.
    4. Measurement in the Pauli‑Z basis and batch‑normalisation.

    The module is compatible with the classical `HybridConv` interface:
    * ``forward`` accepts a tensor of shape (B, 1, H, W) or (H, W).
    * ``kernel_matrix`` is provided for RBF kernel evaluation (classical fallback).
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 127, n_wires: int = 8):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_wires = n_wires

        # Quantum device for batch execution
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Feature map: encode each input pixel with a trainable Ry rotation
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x8_ryzxy"])

        # QCNN‑style convolution and pooling layers
        self.conv_params = tq.ParameterList([tq.Parameter(0.0) for _ in range(self.n_wires * 3)])
        self.pool_params = tq.ParameterList([tq.Parameter(0.0) for _ in range(self.n_wires // 2 * 3)])

        # Fully connected quantum block (Quantum‑NAT style)
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = torch.nn.BatchNorm1d(self.n_wires)

    def conv_circuit(self, qubits: list[int], params: list[torch.Tensor]) -> tq.QuantumCircuit:
        """
        Build a single two‑qubit convolution sub‑circuit.

        Parameters
        ----------
        qubits : list[int]
            Two qubit indices.
        params : list[torch.Tensor]
            Three parameters for the circuit.
        """
        qc = tq.QuantumCircuit(self.n_wires)
        q1, q2 = qubits
        qc.rz(-math.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)
        qc.ry(params[2], q2)
        qc.cx(q2, q1)
        qc.rz(math.pi / 2, q1)
        return qc

    def pool_circuit(self, qubits: list[int], params: list[torch.Tensor]) -> tq.QuantumCircuit:
        """
        Build a two‑qubit pooling sub‑circuit (identical to conv without the final RZ on q1).

        Parameters
        ----------
        qubits : list[int]
            Two qubit indices.
        params : list[torch.Tensor]
            Three parameters for the circuit.
        """
        qc = tq.QuantumCircuit(self.n_wires)
        q1, q2 = qubits
        qc.rz(-math.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)
        qc.ry(params[2], q2)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantum circuit on the input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (H, W).

        Returns
        -------
        torch.Tensor
            Batch‑normalised measurement vector of shape (B, n_wires).
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        bsz, _, H, W = inputs.shape
        # Flatten to a 1‑D feature vector per sample
        flat = inputs.view(bsz, -1)

        # 1. Encode classical data
        self.encoder(self.q_device, flat)

        # 2. QCNN‑style convolution layer
        for i in range(0, self.n_wires, 2):
            params = self.conv_params[i : i + 3]
            self.conv_circuit([i, i + 1], params)(self.q_device)

        # 3. Pooling layer (reduce qubits by half)
        for i in range(0, self.n_wires, 4):
            params = self.pool_params[i // 2 : i // 2 + 3]
            self.pool_circuit([i, i + 2], params)(self.q_device)

        # 4. Quantum‑NAT fully connected block
        self.random_layer(self.q_device)
        self.rx(self.q_device, wires=0)
        self.ry(self.q_device, wires=1)
        self.rz(self.q_device, wires=3)
        self.crx(self.q_device, wires=[0, 2])
        tqf.hadamard(self.q_device, wires=3)
        tqf.sx(self.q_device, wires=2)
        tqf.cnot(self.q_device, wires=[3, 0])

        # 5. Measurement and normalisation
        meas = self.measure(self.q_device)  # shape: (B, n_wires)
        return self.norm(meas)

    @staticmethod
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """
        Classical RBF kernel used as a helper; quantum implementation could replace this.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors.
        gamma : float, optional
            Kernel parameter.

        Returns
        -------
        torch.Tensor
            Scalar kernel value.
        """
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff))

    @staticmethod
    def kernel_matrix(
        a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0
    ) -> np.ndarray:
        """
        Compute a Gram matrix using the RBF kernel (fallback to classical).

        Parameters
        ----------
        a, b : iterable of torch.Tensor
            Collections of 1‑D tensors.
        gamma : float, optional
            Kernel parameter.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        return np.array(
            [[HybridConv.rbf_kernel(x, y, gamma).item() for y in b] for x in a]
        )


def Conv() -> HybridConv:
    """
    Factory function that returns a drop‑in replacement for the original `Conv` filter,
    but executing on a TorchQuantum device.
    """
    return HybridConv()


__all__ = ["HybridConv", "Conv", "HybridConv.kernel_matrix"]
