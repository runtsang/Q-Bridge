"""Quantum hybrid model that replaces the CNN backbone of Quantum‑NAT with a variational
quantum convolutional filter and retains the quantum fully‑connected block.

The model uses Qiskit to implement a small quantum circuit that processes each
4×4 patch of the input image.  The resulting scalars form a 4×4 feature map,
which is then encoded into a 4‑wire quantum device and passed through a
variational quantum layer identical to the one used in the original paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit.circuit.random import random_circuit
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvCircuit:
    """Quantum convolution filter that processes a 4×4 patch and returns a scalar feature."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Encode the data into rotation angles and evaluate the circuit."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = [
            {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            for dat in data
        ]
        job = qiskit.execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class HybridConvModel(tq.QuantumModule):
    """Quantum hybrid model that mirrors the classical HybridConvModel."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, kernel_size: int = 4, threshold: float = 0.5, shots: int = 100):
        super().__init__()
        self.quantum_conv = QuanvCircuit(kernel_size, qiskit.Aer.get_backend("qasm_simulator"), shots, threshold)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Unfold the input into non‑overlapping 4×4 patches
        patches = F.unfold(x, kernel_size=4, stride=4)  # shape: (bsz, 16, 16)
        patches = patches.permute(0, 2, 1)  # (bsz, 16, 16)
        # Apply the quantum convolution to each patch
        conv_feats = torch.empty(bsz, 16, device=x.device)
        for i in range(bsz):
            for j in range(16):
                patch = patches[i, j].cpu().numpy().reshape((4, 4))
                conv_feats[i, j] = self.quantum_conv.run(patch)
        conv_feats = conv_feats.view(bsz, 1, 4, 4)
        # Encode the feature map into a 4‑wire quantum device
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, conv_feats)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridConvModel"]
