"""Quantum‑classical quanvolution network with a trainable variational circuit.

The model implements a variational quantum kernel per 2×2 image patch.  The
encoder maps each pixel to a qubit via Ry rotations, followed by a small
parameterized ansatz shared across all patches.  The resulting expectation
values are concatenated into a feature vector and fed into a two‑layer MLP
classifier.  The class shares the same name `QuanvolutionNet` with the
classical counterpart, enabling interchangeable usage in experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Iterable, Optional

__all__ = ["QuanvolutionNet"]


class ParameterizedAnsatz(tq.QuantumModule):
    """A small trainable circuit applied to each patch."""

    def __init__(self, n_qubits: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Parameter matrix: (depth, n_qubits, 2) for Ry and Rz angles
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 2))
        # Simple entangling pattern: CNOTs between consecutive qubits
        self.cnot_pattern = [(i, i + 1) for i in range(n_qubits - 1)]

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for d in range(self.params.shape[0]):
            for q in range(self.n_qubits):
                theta_y, theta_z = self.params[d, q]
                qdev.ry(theta_y, q)
                qdev.rz(theta_z, q)
            for src, tgt in self.cnot_pattern:
                qdev.cx(src, tgt)


class QuanvolutionNet(tq.QuantumModule):
    """Quantum‑classical quanvolution network."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4  # one qubit per pixel in the 2×2 patch
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.ansatz = ParameterizedAnsatz(n_qubits=self.n_wires, depth=2)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Two‑layer MLP head
        self.classifier = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to (batch, 28, 28) if necessary
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Gather the 2×2 patch
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode the classical data into qubit states
                self.encoder(qdev, patch)
                # Apply the shared variational ansatz
                self.ansatz(qdev)
                # Measure expectation values
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))

        # Concatenate all patch features
        features = torch.cat(patches, dim=1)  # shape: (bsz, 4*14*14)

        # Classical head
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Convenience training helper
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Simple training loop for the hybrid model."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                log_probs = self(xb)
                loss = criterion(log_probs, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            if verbose:
                avg_loss = epoch_loss / len(train_loader.dataset)
                if val_loader is None:
                    print(f"[Epoch {epoch}] loss={avg_loss:.4f}")
                else:
                    val_acc = self._evaluate(val_loader, device)
                    print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    def _evaluate(self, loader: Iterable, device: str) -> float:
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                log_probs = self(xb)
                preds = log_probs.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total
