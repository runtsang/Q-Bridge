"""Quantum utilities for the hybrid fraud‑detection model.

This module implements:
  • Photonic circuit construction mirroring the original seed (Strawberry Fields).
  • A Qiskit EstimatorQNN that can be called from the classical side.
  • A TorchQuantum kernel that replaces the classical RBF kernel.

The functions are lightweight wrappers that expose a consistent API for the ML side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# ------------------------------------------------------------------
#  Photonic (Strawberry Fields) utilities
# ------------------------------------------------------------------
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

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


def build_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Return a Strawberry Fields program mirroring the photonic fraud circuit."""
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


# ------------------------------------------------------------------
#  Qiskit EstimatorQNN utilities
# ------------------------------------------------------------------
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as _EstimatorQNN
from qiskit.primitives import StatevectorEstimator as _Estimator

def build_qiskit_estimator() -> _EstimatorQNN:
    """Return a Qiskit EstimatorQNN that models a simple 1‑qubit circuit."""
    # Define circuit with one trainable weight
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)

    # Observable is Pauli Y
    observable = SparsePauliOp.from_list([("Y", 1)])

    # Build EstimatorQNN
    estimator = _Estimator()
    return _EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator,
    )


# ------------------------------------------------------------------
#  TorchQuantum kernel utilities
# ------------------------------------------------------------------
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data via a fixed list of Ry gates."""
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
    """Quantum kernel evaluated with a 4‑wire TorchQuantum ansatz."""
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
        # Return absolute value of the first element of the state vector
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between two sets of samples."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "FraudLayerParameters",
    "build_photonic_program",
    "build_qiskit_estimator",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]
