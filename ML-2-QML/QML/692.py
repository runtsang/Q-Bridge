"""Quantum Quanvolution module using torchquantum."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class Quanvolution(nn.Module):
    """
    Quantum‑inspired filter that applies a learnable embedding to each
    2×2 image patch, encodes the resulting angles into a small circuit,
    measures all qubits and feeds the concatenated measurements into a
    classical linear head.  Dropout is applied after the measurement
    stage.  The :meth:`train_step` helper enables end‑to‑end optimisation
    with a stochastic optimiser such as Adam.
    """

    def __init__(self, dropout_prob: float = 0.3) -> None:
        super().__init__()
        self.n_wires = 4
        # Learnable mapping from raw pixel values to rotation angles
        self.embedding = nn.Linear(4, 4)
        # Parameterised layer that introduces additional entanglement
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def _encode_patch(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> None:
        """
        Encode a single 2×2 patch into the quantum device.

        Parameters
        ----------
        qdev : tq.QuantumDevice
            Quantum device with a batch dimension.
        patch : torch.Tensor
            Tensor of shape (B, 4) containing raw pixel intensities.
        """
        angles = self.embedding(patch)  # (B, 4)
        for i in range(self.n_wires):
            # Apply Ry rotation with a batch of angles
            qdev.ry(angles[:, i], wires=[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract a 2×2 patch
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self._encode_patch(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)
        features = self.dropout(features)
        logits = self.linear(features)
        return self.logsoftmax(logits)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """
        Execute one training step with the quantum circuit.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, 1, 28, 28).
        y : torch.Tensor
            Target labels of shape (B,).
        optimizer : torch.optim.Optimizer
            Optimiser that will update all learnable parameters.
        loss_fn : nn.Module
            Loss function that accepts logits and targets.

        Returns
        -------
        float
            The scalar loss value.
        """
        self.train()
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["Quanvolution"]
