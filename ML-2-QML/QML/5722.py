from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# ----------------------------------------------------------------------
# Quantum photonic layer definition – identical to the classical seed
# but expressed as a Strawberry Fields program.
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# ----------------------------------------------------------------------
# Quantum kernel – uses TorchQuantum to encode data and compute state
# overlap.  The ansatz mirrors the photonic layer but is expressed in
# discrete‑gate language for compatibility with the QML module.
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Data‑encoding ansatz that maps two input vectors into a shared state."""
    def __init__(self, func_list: Sequence[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Hybrid quantum interface – exposes the same public API as the
# classical counterpart but implements the quantum side.
# ----------------------------------------------------------------------
class HybridFraudDetector:
    """
    Quantum implementation of the hybrid fraud‑detection model.
    Mirrors :class:`HybridFraudDetector` from the classical module.
    The constructor builds the photonic program and attaches a
    quantum kernel that can be queried via :meth:`compute_kernel`.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        # Photonic circuit that will be executed on a simulator or real device
        self.program = build_fraud_detection_program(input_params, layers)
        # Quantum kernel for feature‑map evaluation
        self.kernel = Kernel()

    def simulate(self, inputs: torch.Tensor) -> np.ndarray:
        """Run the photonic program on a batch of classical inputs."""
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        states = []
        for sample in inputs:
            prog = self.program.copy()
            # Encode the sample via displacements – this is a simple illustration
            for i, val in enumerate(sample):
                Dgate(val.item(), 0) | prog.context.wires[i]
            result = eng.run(prog)
            states.append(result.state)
        return np.array(states)

    def compute_kernel(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return a Gram matrix using the quantum kernel."""
        return kernel_matrix(a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that simply evaluates the quantum kernel against
        the input and returns a scalar feature.  In practice this would
        be fed into a classical classifier or another quantum module.
        """
        # Use the kernel against a fixed reference set (here we use the
        # first sample as a placeholder)
        ref = x[0:1]
        return torch.tensor(self.compute_kernel(x, ref).squeeze(), dtype=x.dtype, device=x.device)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridFraudDetector",
]
