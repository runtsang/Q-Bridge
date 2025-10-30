"""Quantum feature extractor for the hybrid quanvolution network.

Encodes each 2×2 patch of an image into a small quantum circuit,
applies a random entangling layer, and measures the Pauli‑Z
expectation on each qubit.  The measurement outcomes form a
feature vector for each patch, which is flattened across all
patches to produce the quantum feature tensor.
"""

from __future__ import annotations

import torchquantum as tq
import torch


class QuantumFeatureExtractor(tq.QuantumModule):
    """Quantum encoder for 2×2 image patches.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch to encode.
    n_layers : int, default 6
        Number of random two‑qubit gates in the entangling layer.
    """
    def __init__(self, kernel_size: int = 2, n_layers: int = 6) -> None:
        super().__init__()
        self.n_wires = kernel_size ** 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )
        self.layer = tq.RandomLayer(n_ops=n_layers, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, 4*14*14).
        """
        batch_size = x.size(0)
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch_size, device=x.device)

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
                )
                self.encoder(qdev, patch)
                self.layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(batch_size, self.n_wires))

        return torch.cat(patches, dim=1)


def get_quantum_filter() -> QuantumFeatureExtractor:
    """Convenience factory that returns a ready‑to‑use quantum feature extractor."""
    return QuantumFeatureExtractor()


__all__ = ["QuantumFeatureExtractor", "get_quantum_filter"]
