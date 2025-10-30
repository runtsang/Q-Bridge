"""Quantum variational model that mirrors the classical hybrid architecture.

The implementation uses torchquantum to build a parameterised circuit
whose structure is inspired by the QCNN feature‑map, the QuantumNAT
quantum layer, and the classical fully‑connected block.  The model
exposes an ``evaluate`` method that mimics the FastBaseEstimator
behaviour, optionally adding Gaussian shot noise.

Key components
---------------
* 4‑wire encoder that embeds a 4‑pixel patch (mimicking the 4‑pixel
  pooling used in the classical model)
* ``QuantumBlock`` that applies a random layer followed by
  trainable single‑ and two‑qubit gates, similar to the QFCModel
  quantum layer
* Classical post‑processing fully‑connected network
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from typing import Iterable, List, Callable

class SharedClassName(tq.QuantumModule):
    """Quantum variational model mirroring the hybrid classical design."""

    class QuantumBlock(tq.QuantumModule):
        """Parameterised block with random and trainable gates."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4, n_classes: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps a 4‑pixel patch to a 4‑wire state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.quantum_block = self.QuantumBlock(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)
        # Classical post‑processing (equivalent to the FC block)
        self.fc = nn.Sequential(
            nn.Linear(n_wires, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum circuit and classical head."""
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pool 6×6 patches from input image to match the classical 4‑pixel
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.quantum_block(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        out = self.fc(out)
        return out

    def evaluate(
        self,
        inputs: Iterable[torch.Tensor],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]] = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the quantum model on a collection of inputs, optionally
        adding Gaussian shot noise.

        Parameters
        ----------
        inputs : iterable of tensors
            Each element is a single sample (batch dimension is added
            automatically).
        observables : iterable of callables
            Functions that map the model output to a scalar.  If
            ``None`` a single observable returning the mean over the
            output dimension is used.
        shots : int, optional
            Number of simulated shots.  If ``None`` no noise is added.
        seed : int, optional
            Random seed for reproducible shot noise.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for inp in inputs:
                out = self.forward(inp.unsqueeze(0))
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["SharedClassName"]
