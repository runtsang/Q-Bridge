from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QuantumRBFAnsatz(tq.QuantumModule):
    """
    Simple parameter‑encoded ansatz that applies Ry rotations to each wire.
    Parameters are expected to be a 1‑D tensor of length n_wires.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, params: torch.Tensor) -> None:
        q_device.reset_states(params.shape[0])
        for i in range(self.n_wires):
            tq.ry(q_device, params[:, i], wires=[i])


class ClassicalRBF(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernelMethod(tq.QuantumModule):
    """
    Quantum‑classical hybrid kernel implemented with TorchQuantum.
    Allows optional classical feature extraction and weighted combination
    of a classical RBF kernel and a quantum‑encoded kernel.
    """

    def __init__(
        self,
        n_wires: int = 4,
        feature_extractor: Optional[nn.Module] = None,
        quantum_ansatz: Optional[tq.QuantumModule] = None,
        weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.feature_extractor = feature_extractor
        self.quantum_ansatz = quantum_ansatz or QuantumRBFAnsatz(n_wires)
        self.classical_kernel = ClassicalRBF(gamma=1.0)
        self.weight = float(weight)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x) if self.feature_extractor else x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._extract(x).reshape(1, -1)
        y = self._extract(y).reshape(1, -1)
        classical = self.classical_kernel(x, y).squeeze(-1)
        self.quantum_ansatz(self.q_device, x)
        self.quantum_ansatz(self.q_device, -y)
        quantum = torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(-1)
        return self.weight * classical + (1.0 - self.weight) * quantum

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        self.eval()
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


class FastHybridEstimator:
    """
    Evaluator that runs the hybrid kernel on a batch of parameter sets
    with optional shot noise to emulate measurement statistics.
    """

    def __init__(self, kernel: HybridKernelMethod) -> None:
        self.kernel = kernel

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        self.kernel.eval()
        with torch.no_grad():
            for params in parameter_sets:
                x = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                y = x.clone()
                val = self.kernel(x, y).item()
                if shots is None:
                    results.append([val])
                else:
                    noisy = rng.normal(val, max(1e-6, 1.0 / shots))
                    results.append([float(noisy)])
        return results


def SamplerQNN() -> tq.QuantumModule:
    """
    Simple parameterised quantum circuit that can be used as a sampler.
    """
    class SamplerCircuit(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 2
            self.params = torch.nn.Parameter(torch.zeros(4))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, params: torch.Tensor) -> None:
            q_device.reset_states(params.shape[0])
            tq.ry(q_device, params[:, 0], wires=[0])
            tq.ry(q_device, params[:, 1], wires=[1])
            tq.cx(q_device, wires=[0, 1])
            tq.ry(q_device, params[:, 2], wires=[0])
            tq.ry(q_device, params[:, 3], wires=[1])

    return SamplerCircuit()


__all__ = [
    "FraudLayerParameters",
    "QuantumRBFAnsatz",
    "HybridKernelMethod",
    "FastHybridEstimator",
    "SamplerQNN",
]
