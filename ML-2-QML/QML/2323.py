"""Hybrid quantum network mirroring :mod:`Quanvolution__gen127.py`.

The quantum implementation uses :mod:`torchquantum` to build a
variational quanvolution filter and a parameterised sampler.
Both the filter and the sampler are exposed through a single
``QuanvolutionHybrid`` class that can be switched between a
classical linear head and a quantum sampler head.

Classes
-------
QuanvolutionFilter
    Variational 2‑D filter that applies a random two‑qubit kernel to 2×2 patches.
SamplerQNN
    Parameterised quantum circuit that encodes a 4‑dimensional input and returns
    a 10‑dimensional probability vector.
QuanvolutionClassifier
    Quantum classifier that stacks the filter with a linear head.
QuanvolutionHybrid
    Flexible wrapper that can use either the linear head or the sampler head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class SamplerQNN(tq.QuantumModule):
    """Parameterised quantum sampler that outputs a 10‑dimensional probability vector."""

    def __init__(self, input_dim: int = 4, n_wires: int = 10) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_wires = n_wires
        # Encode the first ``input_dim`` wires using Ry rotations.
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(input_dim)]
        )
        # Random variational layer to mix the qubits.
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 4)`` – the reduced feature vector from the filter.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 10)`` – probability vector for the 10 classes.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # Only the first ``input_dim`` wires are encoded.
        self.encoder(qdev, x[:, : self.input_dim])
        self.q_layer(qdev)
        measurement = self.measure(qdev)  # (batch, n_wires)
        # Convert measurement outcomes from {-1, 1} to probabilities in [0, 1].
        probs = (measurement + 1) / 2
        return probs


class QuanvolutionClassifier(tq.QuantumModule):
    """Quantum classifier using the quanvolution filter followed by a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Flexible hybrid quantum network that can use either a linear head or a quantum sampler head.

    Parameters
    ----------
    use_sampler : bool, optional
        If ``True`` the network will use :class:`SamplerQNN` as the classification head.
        Otherwise a simple linear layer is used.  The default is ``False``.
    """

    def __init__(self, use_sampler: bool = False) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.use_sampler = use_sampler
        if use_sampler:
            # Classical linear layer to reduce the high‑dimensional feature map to 4.
            self.pre_sampler = nn.Linear(4 * 14 * 14, 4)
            self.sampler = SamplerQNN()
        else:
            self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        if self.use_sampler:
            reduced = self.pre_sampler(features)
            logits = self.sampler(reduced)
        else:
            logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "QuanvolutionFilter",
    "SamplerQNN",
    "QuanvolutionClassifier",
    "QuanvolutionHybrid",
]
