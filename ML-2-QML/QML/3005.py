"""Hybrid quantum neural network for the Quantum‑NAT family.

The model embeds 2‑D image features into a 4‑qubit circuit, applies a
variational layer, and measures all qubits in the Pauli‑Z basis.
An estimator wrapper mirrors the classical FastEstimator and adds
shot‑noise to emulate realistic quantum measurements.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridNATModel(tq.QuantumModule):
    """Quantum‑NAT model with a 4‑qubit variational circuit."""

    class QLayer(tq.QuantumModule):
        """Variational layer used after feature encoding."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a 16‑dimensional vector into 4 qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input, run the variational layer, and return
        the normalized measurement results.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Global average pooling to 16 features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class FastBaseEstimator:
    """Deterministic estimator for a tq.QuantumModule.

    Parameters
    ----------
    circuit : tq.QuantumModule
        The quantum model to evaluate.
    """

    def __init__(self, circuit: tq.QuantumModule) -> None:
        self._circuit = circuit

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        """Compute expectation values for each observable and parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        for _ in parameter_sets:
            dummy = torch.zeros(1, 1, 1, 1, device=torch.device('cpu'))
            outputs = self._circuit.forward(dummy)
            row: List[float] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Estimator that adds Gaussian shot noise to quantum outputs.

    The noise model emulates measurement uncertainty proportional to
    1/shots.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridNATModel", "FastBaseEstimator", "FastEstimator"]
