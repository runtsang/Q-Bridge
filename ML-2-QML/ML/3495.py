import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from collections.abc import Iterable, Sequence
from typing import Callable, List
import numpy as np

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of values into a batched tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class QFCModel(tq.QuantumModule):
    """Hybrid classical–quantum network:
    * 2D convolutional encoder followed by a fully‑connected projection.
    * Classical features are encoded into a 4‑wire quantum device via a parametric encoder.
    * A small variational quantum layer processes the encoded state.
    * Output is a batch of 4‑dimensional feature vectors."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
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
            self.crx(qdev, wires=[0, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        # Classical backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        # Quantum encoder & layer
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: classical feature extraction -> quantum encoding -> quantum layer -> measurement."""
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        classical_out = self.fc(flattened)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        encoded = classical_out[:, :self.n_wires]
        self.encoder(qdev, encoded)
        self.q_layer(qdev)
        quantum_out = self.measure(qdev)
        out = quantum_out + classical_out
        return self.norm(out)

class FastEstimator:
    """Lightweight estimator for QFCModel that evaluates observables and optionally adds Gaussian shot noise."""
    def __init__(self, model: QFCModel):
        self.model = model

    def _eval_batch(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None
    ) -> List[List[float]]:
        """For each set of parameters (input images), compute the model output and apply each observable.
        If `shots` is given, add Gaussian noise with standard deviation 1/√shots."""
        results: List[List[float]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            outputs = self._eval_batch(inputs)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy
