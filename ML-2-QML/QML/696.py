"""Quanvolutional filter with a variational quantum circuit.

The module defines two classes:
- QuanvolutionFilter: applies a quantum kernel to 2×2 image patches.
  The kernel consists of a random layer followed by a parameterized
  depth‑controlled variational circuit.  The measurement basis is
  configurable.
- QuanvolutionClassifier: a hybrid network that stacks the filter
  and a linear head.  The interface matches the classical version, so
  existing training scripts can be reused.

The design keeps the original API while adding quantum expressivity
and the ability to train the variational parameters end‑to‑end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that maps 2×2 patches to 4‑dimensional vectors.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits per patch.
    depth : int, default 3
        Depth of the parameterized variational circuit.
    measurement_basis : str, default 'Z'
        Basis used for the final measurement ('Z' or 'X').
    """

    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 3,
        measurement_basis: str = "Z",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires

        # Simple Ry encoder that maps pixel values to rotation angles.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Random layer for added expressivity.
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))

        # Parameterized variational layer with tunable depth.
        self.param_layer = tq.QuantumCircuit()
        for d in range(depth):
            for w in range(n_wires):
                self.param_layer += tq.RZ(
                    wires=[w], params=[tq.Param(f"theta_{d}_{w}")]
                )

        # Measurement in the chosen basis.
        self.measure = tq.MeasureAll(
            tq.PauliZ if measurement_basis.upper() == "Z" else tq.PauliX
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply the quantum filter to a batch of images."""
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to 28×28 per image and iterate over 2×2 patches.
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                self.param_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the quantum filter followed by a linear head.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits per patch.
    depth : int, default 3
        Depth of the variational circuit in the filter.
    measurement_basis : str, default 'Z'
        Basis used for the final measurement in the filter.
    num_classes : int, default 10
        Number of target classes.
    """

    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 3,
        measurement_basis: str = "Z",
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            n_wires=n_wires, depth=depth, measurement_basis=measurement_basis
        )
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass."""
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
