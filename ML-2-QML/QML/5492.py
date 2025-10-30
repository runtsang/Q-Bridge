"""Quantum hybrid kernel method using TorchQuantum ansatz and a classical fraud‑detection head."""

from __future__ import annotations

from typing import Sequence, Iterable

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum.operators import op_name_dict

# ----------------------------------------------------------------------
# Classical fraud detection head (re‑used from the ML side)
# ----------------------------------------------------------------------
class FraudLayerParameters:
    def __init__(self, bs_theta: float, bs_phi: float,
                 phases: tuple[float, float], squeeze_r: tuple[float, float],
                 squeeze_phi: tuple[float, float], displacement_r: tuple[float, float],
                 displacement_phi: tuple[float, float], kerr: tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> tq.QuantumModule:
    class SimpleLinear(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.params = params

        def forward(self, q_device: tq.QuantumDevice) -> None:
            tq.RY(q_device, w=0, theta=self.params.bs_theta)
    return SimpleLinear()

def build_fraud_detection_program(params: FraudLayerParameters) -> tq.QuantumModule:
    return _layer_from_params(params, clip=False)

# ----------------------------------------------------------------------
# Quantum kernel
# ----------------------------------------------------------------------
class HybridQuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel module that evaluates a feature‑map overlap.

    The circuit first applies a self‑attention style rotation block followed
    by a controlled‑rotation entanglement pattern.  The overlap of the
    resulting states is used as the kernel value.
    """

    def __init__(self, embed_dim: int = 4, n_wires: int | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires or embed_dim
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> list[dict]:
        ansatz = []
        for i in range(self.n_wires):
            ansatz.append({"input_idx": [i], "func": "rx", "wires": [i]})
            ansatz.append({"input_idx": [i], "func": "ry", "wires": [i]})
            ansatz.append({"input_idx": [i], "func": "rz", "wires": [i]})
        for i in range(self.n_wires - 1):
            ansatz.append({"input_idx": [i], "func": "crx", "wires": [i, i + 1]})
        return ansatz

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel_value(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Hybrid quantum + classical fraud detection head
# ----------------------------------------------------------------------
class HybridQuantumKernelMethodWithClassicalHead(HybridQuantumKernelMethod):
    """Wraps the quantum kernel with a classical fraud‑detection head."""

    def __init__(self, embed_dim: int = 4,
                 fraud_params: FraudLayerParameters | None = None) -> None:
        super().__init__(embed_dim=embed_dim)
        self.fraud_head = build_fraud_detection_program(fraud_params or
            FraudLayerParameters(0.0, 0.0, (0.0, 0.0), (0.0, 0.0),
                                  (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.eye(self.n_wires)
        features = torch.stack([self.kernel_value(x, b) for b in basis])
        return self.fraud_head(features)

__all__ = ["HybridQuantumKernelMethod", "HybridQuantumKernelMethodWithClassicalHead",
           "FraudLayerParameters", "build_fraud_detection_program"]
