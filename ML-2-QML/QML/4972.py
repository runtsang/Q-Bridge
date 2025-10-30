import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(val: float, bound: float) -> float:
    return max(-bound, min(bound, val))

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum convolutional filter that mirrors the classical architecture
    but uses a random layered circuit on each 2x2 image patch.  The module
    is followed by a classical linear classifier.  The implementation
    incorporates fraud‑detection style clipping and supports shot‑noise
    evaluation similar to FastBaseEstimator.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        n_wires: int = 4,
        n_ops: int = 8,
        threshold: float = 0.5,
        clip: bool = True,
        clip_bound: float = 5.0,
        classifier_units: int = 10,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_wires = n_wires
        self.threshold = threshold
        self.clip = clip
        self.clip_bound = clip_bound

        # Quantum encoder that maps a 2x2 patch to qubit amplitudes
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear head
        self.linear = nn.Linear(
            in_features=n_wires * ((28 - kernel_size) // kernel_size + 1) ** 2,
            out_features=classifier_units,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter to each image patch and classify."""
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape image to patches
        patches = x.view(bsz, 28, 28)
        measurements = []
        for r in range(0, 28, self.kernel_size):
            for c in range(0, 28, self.kernel_size):
                patch = torch.stack(
                    [
                        patches[:, r, c],
                        patches[:, r, c + 1],
                        patches[:, r + 1, c],
                        patches[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev)
                # Convert Pauli‑Z outcomes to mean value in [0,1]
                meas = (meas + 1) / 2
                measurements.append(meas)
        # Concatenate all patch measurements
        features = torch.cat(measurements, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Evaluation utilities (shot‑noise aware)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set with optional shot noise."""
        if not observables:
            observables = [lambda out: out.mean()]

        results: List[List[complex]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                self._load_params(params)
                dummy = torch.zeros(1, 1, 28, 28, device="cpu")
                output = self(dummy)
                row: List[complex] = []
                for obs in observables:
                    val = obs(output).item()
                    row.append(val)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [float(rng.normal(val.real, max(1e-6, 1 / shots))) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                         for val in row]
            noisy.append(noisy_row)
        return noisy

    def _load_params(self, flat_params: Sequence[float]) -> None:
        """Load flattened parameters into the linear classifier."""
        params = torch.as_tensor(flat_params, dtype=torch.float32)
        lin_params = params[: self.linear.numel()]
        lin_weights = lin_params[: self.linear.weight.numel()]
        lin_bias = lin_params[self.linear.weight.numel() :]
        self.linear.weight.data = lin_weights.reshape(self.linear.weight.shape)
        self.linear.bias.data = lin_bias.reshape(self.linear.bias.shape)

__all__ = ["QuanvolutionHybrid"]
