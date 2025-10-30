"""Quantum‑enhanced quanvolution filter with a learnable variational kernel.

Each 2×2 patch is encoded into four qubits, passed through a depth‑controlled
variational circuit, and measured. The resulting classical features are
concatenated and fed into a classical classifier head with a residual
shortcut for better expressivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class Quanvolution__gen211(tq.QuantumModule):
    """Quantum‑enhanced quanvolution neural network."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_out_channels: int = 4,
        depth: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv_out_channels = conv_out_channels
        self.depth = depth
        self.device = device

        # Encoder: map pixel values to rotation angles on four qubits.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Variational circuit: depth layers of parameterised rotations and CNOTs
        self.var_circuit = tq.QuantumCircuit(4, circuit_name="var")
        for d in range(depth):
            for w in range(4):
                self.var_circuit += tq.Rotations(
                    "ry", wires=[w], parameters=[self.add_parameter(f"ry_{w}_{d}")]
                )
            # Entanglement pattern: linear chain
            for w in range(3):
                self.var_circuit += tq.CNOT(wires=[w, w + 1])

        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head: two‑layer MLP with residual
        self.fc1 = nn.Linear(4 * 14 * 14, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        batch_size = x.shape[0]
        device = x.device

        # Prepare patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # (batch, 4)
                patches.append(patch)

        n_patches = len(patches)  # 14*14 = 196
        qdev = tq.QuantumDevice(4, bsz=batch_size, device=device)

        outputs = []

        for patch in patches:
            self.encoder(qdev, patch)
            qdev.apply_circuit(self.var_circuit)
            out = self.measure(qdev)  # (batch, 4)
            outputs.append(out)

        # Concatenate all patch outputs
        flat = torch.cat(outputs, dim=1)  # (batch, 4 * n_patches)

        h = F.relu(self.bn1(self.fc1(flat)))
        logits = self.fc2(h)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen211"]
