"""Hybrid quantum fraud detection model combining photonic circuit and quantum kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# ----------------------------------------------------------------------
# Photonic circuit construction
# ----------------------------------------------------------------------
def build_fraud_detection_program(
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# ----------------------------------------------------------------------
# Quantum kernel construction using TorchQuantum
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
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

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
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

# ----------------------------------------------------------------------
# Hybrid model exposing both photonic program and quantum kernel
# ----------------------------------------------------------------------
class FraudDetectionHybrid:
    """Hybrid model that can be used with either a photonic circuit or a quantum kernel."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        support_vectors: Sequence[torch.Tensor] | None = None,
    ) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.kernel = Kernel()
        self.support_vectors = support_vectors

    def simulate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Placeholder for photonic simulation – returns zero tensor."""
        return torch.tensor(0.0)

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the quantum kernel matrix between two sets of samples."""
        if y is None:
            y = x
        return self.kernel(x, y)

    def similarity_to_support(self, x: torch.Tensor) -> torch.Tensor:
        """Compute similarities between `x` and a set of support vectors."""
        if self.support_vectors is None:
            raise ValueError("Support vectors not provided")
        return torch.stack([self.kernel_matrix(x, sv) for sv in self.support_vectors], dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid", "Kernel"]
