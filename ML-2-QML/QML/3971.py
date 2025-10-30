"""Hybrid kernel and fraud detection module – quantum implementation.

This module combines:
- a quantum kernel evaluated with a fixed TorchQuantum ansatz,
- a photonic fraud‑detection circuit built with Strawberry Fields,
- and a convenient interface for building and evaluating both.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from strawberryfields.backends.gaussian import Sim
from typing import Sequence, Iterable
from dataclasses import dataclass

# -------------------- Quantum kernel primitives --------------------
class QuantumAnsatz(tq.QuantumModule):
    """Encodes classical data through a list of parameterised gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# -------------------- Photonic fraud detection primitives --------------------
@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip_value(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip_value(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip_value(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip_value(k, 1)) | modes[i]

def build_photonic_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# -------------------- High‑level hybrid class --------------------
class HybridKernelFraudDetector:
    """Unified interface for quantum kernels and photonic fraud detection.

    The class exposes:
      * `kernel_matrix(a, b)` – quantum kernel Gram matrix.
      * `photonic_program(input_params, layers)` – builds a Strawberry Fields program.
      * `run_photonic(program, backend)` – executes the program on a chosen backend.
    """
    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return quantum_kernel_matrix(a, b)

    def photonic_program(self, input_params: FraudLayerParameters,
                         layers: Iterable[FraudLayerParameters]) -> sf.Program:
        return build_photonic_fraud_detection_program(input_params, layers)

    def run_photonic(self, program: sf.Program, backend: str = "gaussian") -> np.ndarray:
        """Run the Strawberry Fields program on the requested backend."""
        if backend == "gaussian":
            sim = Sim()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        result = sim.run(program)
        return result.state
