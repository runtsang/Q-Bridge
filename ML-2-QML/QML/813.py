"""Quantum‑enhanced model with a multi‑layer entangled variational circuit based on Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum model that encodes classical data into a 4‑qubit circuit and applies a deep variational ansatz."""

    class VariationalLayer(tq.QuantumModule):
        """Depth‑controlled variational layer with parameterised rotations and CNOT entanglement."""

        def __init__(self, n_wires: int, depth: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth

            # Build a list of rotation layers
            self.layers = nn.ModuleList()
            for _ in range(depth):
                layer = nn.ModuleDict()
                for i in range(n_wires):
                    layer[f"rx_{i}"] = tq.RX(has_params=True, trainable=True)
                    layer[f"ry_{i}"] = tq.RY(has_params=True, trainable=True)
                    layer[f"rz_{i}"] = tq.RZ(has_params=True, trainable=True)
                self.layers.append(layer)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in self.layers:
                # Apply rotations
                for name, op in layer.items():
                    wire = int(name.split("_")[-1])
                    op(qdev, wires=wire)
                # Entangle all neighbouring qubits
                for i in range(self.n_wires - 1):
                    tq.CNOT(qdev, wires=[i, i + 1])
            # Final Hadamard on all wires for better state mixing
            tqf.hadamard(qdev, wires=list(range(self.n_wires)), static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

        # Classical encoder that maps a 16‑dimensional vector to rotation angles
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Variational ansatz
        self.var_layer = self.VariationalLayer(self.n_wires, depth=4)

        # Measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical post‑processing head
        self.fc = nn.Sequential(
            nn.Linear(self.n_wires, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )

        # Encode the input: average‑pool to 16 values, then map to angles
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Apply the variational circuit
        self.var_layer(qdev)

        # Measure expectation values
        out = self.measure(qdev)

        # Classical head
        out = self.fc(out)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
