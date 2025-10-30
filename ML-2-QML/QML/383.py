"""Hybrid network: classical conv + variational quantum filter + skip‑connection + linear head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionPlus(nn.Module):
    """Hybrid network: classical conv + variational quantum filter + skip‑connection + linear head."""

    def __init__(self) -> None:
        super().__init__()
        # classical path
        self.classical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.classical_bn = nn.BatchNorm2d(4)
        self.classical_relu = nn.ReLU(inplace=True)
        # quantum device
        self.dev = qml.device("default.qubit", wires=2)
        # variational parameters
        self.var_params = nn.Parameter(torch.randn(4))
        # quantum node
        self.qnode = qml.QNode(self._quantum_circuit, device=self.dev, interface="torch")
        # linear head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def _quantum_circuit(self, patch: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that encodes a 2x2 patch into a 4‑dimensional feature vector."""
        qml.RY(patch[0], wires=0)
        qml.RY(patch[1], wires=1)
        qml.RY(patch[2], wires=0)
        qml.RY(patch[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(self.var_params[0], wires=0)
        qml.RY(self.var_params[1], wires=1)
        qml.CNOT(wires=[1, 0])
        qml.RY(self.var_params[2], wires=0)
        qml.RY(self.var_params[3], wires=1)
        return torch.stack(
            [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliX(1)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out = self.classical_conv(x)
        cls_out = self.classical_bn(cls_out)
        cls_out = self.classical_relu(cls_out)
        cls_out = cls_out.view(x.size(0), -1)
        bsz = x.shape[0]
        patch_features = []
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
                )
                q_out = torch.stack([self.qnode(patch[i]) for i in range(bsz)], dim=0)
                patch_features.append(q_out)
        q_out = torch.cat(patch_features, dim=1)
        fused = cls_out + q_out
        logits = self.linear(fused)
        return F.log_softmax(logits, dim=-1)
