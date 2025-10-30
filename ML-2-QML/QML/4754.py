"""Hybrid quantum architecture combining a quantum feature encoder, variational QLayer, quantum autoencoder, and QCNN‑style quantum layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumModel(tq.QuantumModule):
    """
    Quantum hybrid network that mirrors the classical pipeline:
      1. Encodes a 2‑D image into a 4‑qubit quantum state.
      2. Applies a variational QLayer (random + trainable ops).
      3. Runs a quantum autoencoder based on a swap‑test subcircuit.
      4. Executes QCNN‑style convolution and pooling layers on qubits.
      5. Measures all qubits and normalises the result.
    """
    class QLayer(tq.QuantumModule):
        """Variational layer mixing random and trainable single‑qubit rotations."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
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
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    class QuantumAutoEncoder(tq.QuantumModule):
        """Swap‑test based quantum autoencoder with 2 trash qubits."""
        def __init__(self, latent: int, trash: int) -> None:
            super().__init__()
            self.latent = latent
            self.trash = trash
            self.ansatz = tq.RealAmplitudes(num_qubits=latent + trash, reps=3)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Encode data onto latent + trash qubits
            self.ansatz(qdev, wires=list(range(self.latent + self.trash)))
            # Swap‑test with auxiliary qubit
            aux = self.latent + 2 * self.trash
            tqf.hadamard(qdev, wires=aux, static=self.static_mode, parent_graph=self.graph)
            for i in range(self.trash):
                tqf.cswap(qdev, wires=[aux, self.latent + i, self.latent + self.trash + i],
                          static=self.static_mode, parent_graph=self.graph)
            tqf.hadamard(qdev, wires=aux, static=self.static_mode, parent_graph=self.graph)
            # Measurement happens later in the main forward

    class QCNNLayers(tq.QuantumModule):
        """QCNN‑style convolution and pooling implemented with parameterised gates."""
        def __init__(self, num_qubits: int) -> None:
            super().__init__()
            self.num_qubits = num_qubits
            # Convolution block: 3‑parameter per pair
            self.conv_params = tq.ParameterVector("θ_conv", length=num_qubits // 2 * 3)
            # Pooling block: 3‑parameter per pair
            self.pool_params = tq.ParameterVector("θ_pool", length=num_qubits // 2 * 3)

        @tq.static_support
        def conv_layer(self, qdev: tq.QuantumDevice, params: tq.ParameterVector, qubits: list[int]) -> None:
            for i in range(0, len(qubits), 2):
                idx = i // 2 * 3
                tqf.rz(qdev, params[idx], wires=qubits[i], static=self.static_mode, parent_graph=self.graph)
                tqf.ry(qdev, params[idx + 1], wires=qubits[i + 1], static=self.static_mode, parent_graph=self.graph)
                tqf.cx(qdev, wires=[qubits[i], qubits[i + 1]], static=self.static_mode, parent_graph=self.graph)

        @tq.static_support
        def pool_layer(self, qdev: tq.QuantumDevice, params: tq.ParameterVector, qubits: list[int]) -> None:
            for i in range(0, len(qubits), 2):
                idx = i // 2 * 3
                tqf.rz(qdev, params[idx], wires=qubits[i], static=self.static_mode, parent_graph=self.graph)
                tqf.ry(qdev, params[idx + 1], wires=qubits[i + 1], static=self.static_mode, parent_graph=self.graph)
                tqf.cx(qdev, wires=[qubits[i], qubits[i + 1]], static=self.static_mode, parent_graph=self.graph)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # First convolution on all qubits
            self.conv_layer(qdev, self.conv_params, list(range(self.num_qubits)))
            # Pool to half the qubits
            self.pool_layer(qdev, self.pool_params, list(range(self.num_qubits // 2)))

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 8   # 4 for data + 4 for QCNN
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x8_ryzxy"])
        self.q_layer = self.QLayer()
        self.autoenc = self.QuantumAutoEncoder(latent=4, trash=2)
        self.qcnn = self.QCNNLayers(num_qubits=4)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # 1. Feature pooling then encoding
        pooled = torch.mean(x, dim=[2, 3])  # global average pooling
        self.encoder(qdev, pooled)

        # 2. Variational QLayer
        self.q_layer(qdev)

        # 3. Quantum autoencoder on first 4 qubits
        self.autoenc(qdev)

        # 4. QCNN layers on remaining 4 qubits
        self.qcnn(qdev)

        # 5. Measurement and normalisation
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridQuantumModel"]
