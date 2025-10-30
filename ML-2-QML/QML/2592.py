"""Quantum implementation of fraud detection kernel and photonic circuit.

The module defines:
  * `QuantumFraudKernel` – a TorchQuantum module that encodes two classical
    vectors into a quantum state and returns the absolute overlap.
  * `FraudDetectionHybrid` – a convenience wrapper that exposes the same
    interface as the classical `FraudDetectionHybrid` but operates purely
    on quantum states.
  * `PhotonicFraudCircuit` – a Strawberry Fields program that implements the
    photonic fraud detection circuit described in the original seed.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from typing import Sequence, Tuple

class QuantumFraudKernel(tq.QuantumModule):
    """Quantum kernel that encodes two classical vectors and returns overlap."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Define a simple ansatz of Ry rotations on each wire
        self.ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap between encoded states."""
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class FraudDetectionHybrid(tq.QuantumModule):
    """Quantum wrapper that mimics the classical FraudDetectionHybrid interface."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.kernel = QuantumFraudKernel(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel.kernel(x, y)

class PhotonicFraudCircuit:
    """Builds a Strawberry Fields program for fraud detection."""
    def __init__(self, input_params, layers):
        self.program = sf.Program(2)
        with self.program.context as q:
            self._apply_layer(q, input_params, clip=False)
            for layer in layers:
                self._apply_layer(q, layer, clip=True)

    def _apply_layer(self, modes: Sequence, params, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | modes[i]

    def run(self, backend=None):
        if backend is None:
            backend = sf.Simulator()
        return backend.run(self.program).state

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

__all__ = [
    "QuantumFraudKernel",
    "FraudDetectionHybrid",
    "PhotonicFraudCircuit",
]
