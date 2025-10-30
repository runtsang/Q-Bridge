"""HybridQNAT: a hybrid quantum‑classical model that merges a quantum convolution
layer with a quantum fully‑connected block.

The quantum convolution uses a small Qiskit circuit to process 2×2 patches of the
input image, producing a scalar per patch. The quantum fully‑connected block
is built with TorchQuantum and mirrors the structure of QuantumNAT's QLayer.
The two quantum outputs are concatenated and fed into a classical linear
projection, followed by batch‑norm.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumConv(tq.QuantumModule):
    """2×2 quantum convolution implemented with a Qiskit circuit.

    The circuit applies a trainable RX rotation to each qubit, a small random
    layer, and measures all qubits in the computational basis. The output
    is the average probability of measuring |1> across the qubits.
    """

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the Qiskit circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten and bind parameters
        param_binds = []
        for i, val in enumerate(data.flatten()):
            bind = {self.theta[i]: np.pi if val > self.threshold else 0}
            param_binds.append(bind)

        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        counts = job.result().get_counts(self.circuit)

        # Compute average number of |1> outcomes
        total_ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
        return total_ones / (self.shots * self.n_qubits)


class HybridQNAT(tq.QuantumModule):
    """Hybrid quantum‑classical model inspired by Conv and QuantumNAT.

    The forward pass consists of:
        1. Classical pooling of the input image.
        2. Quantum embedding via a 4‑wire general encoder.
        3. A quantum fully‑connected block (QLayer).
        4. A separate quantum convolution (QuantumConv) applied to the raw image.
        5. Concatenation of the two quantum outputs and a classical linear head.
    """

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

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.quantum_conv = QuantumConv(kernel_size=2, shots=100, threshold=127)

        # Classical linear head that takes the concatenated quantum outputs
        self.fc = nn.Sequential(
            nn.Linear(self.n_wires + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, H, W) with H=W=28.

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional output per sample.
        """
        bsz = x.shape[0]
        # 1. Classical pooling to match QuantumNAT's pooling size
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)  # shape (B, 16)

        # 2. Quantum embedding
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        q_out = self.measure(qdev)  # shape (B, 4)

        # 3. Quantum convolution on raw image patches
        conv_outs = []
        for i in range(bsz):
            # Extract a single 2×2 patch from the top‑left corner of the image
            patch = x[i, 0, :2, :2].cpu().numpy()
            conv_outs.append(self.quantum_conv.run(patch))
        conv_out = torch.tensor(conv_outs, device=x.device).unsqueeze(-1)  # shape (B, 1)

        # 4. Concatenate quantum outputs
        concat = torch.cat([q_out, conv_out], dim=1)  # shape (B, 5)

        # 5. Classical head
        out = self.fc(concat)
        return self.norm(out)


__all__ = ["HybridQNAT"]
