"""
HybridEstimator – Quantum implementation.

Provides a quantum‑centric estimator that can evaluate
Qiskit circuits, Strawberry Fields programs, or TorchQuantum
kernels.  It supports shot‑noise simulation and exposes
factory helpers for fraud‑detection, a variational classifier,
and a fixed quantum kernel.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterable, List, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Imports for the different backends
# --------------------------------------------------------------------------- #

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector, SparsePauliOp
except ImportError:
    QuantumCircuit = None  # type: ignore[assignment]

try:
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
except ImportError:
    sf = None  # type: ignore[assignment]

try:
    import torchquantum as tq
    from torchquantum.functional import func_name_dict
except ImportError:
    tq = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Quantum estimator
# --------------------------------------------------------------------------- #

class HybridEstimator:
    """
    Evaluate a quantum circuit or a quantum kernel.

    Parameters
    ----------
    backend : Union[QuantumCircuit, sf.Program, tq.QuantumModule]
        The quantum object to evaluate.  One of the three backends must be
        available at runtime.
    shots : int, optional
        If supplied, Gaussian shot noise with variance 1/shots is added.
    """

    def __init__(
        self,
        backend: Union["QuantumCircuit", "sf.Program", "tq.QuantumModule"],
        *,
        shots: int | None = None,
    ) -> None:
        self.backend = backend
        self.shots = shots

        # Detect backend type
        if isinstance(backend, QuantumCircuit):
            self._type = "qiskit"
            self._parameters = list(backend.parameters)
        elif sf is not None and isinstance(backend, sf.Program):
            self._type = "sf"
        elif tq is not None and isinstance(backend, tq.QuantumModule):
            self._type = "tq"
            self.q_device = backend.q_device
        else:
            raise TypeError("Unsupported quantum backend provided.")

    # --------------------------------------------------------------------- #
    # Helper for binding parameters (Qiskit)
    # --------------------------------------------------------------------- #

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if not isinstance(self.backend, QuantumCircuit):
            raise RuntimeError("Binding only supported for Qiskit circuits.")
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.backend.assign_parameters(mapping, inplace=False)

    # --------------------------------------------------------------------- #
    # Evaluation
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        observables: Iterable[Union["BaseOperator", "SparsePauliOp"]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable
            For Qiskit: BaseOperator instances.  For SF: any operator
            that can be passed to Statevector.expectation_value.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one batch.
        """
        results: List[List[complex]] = []

        if self._type == "qiskit":
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)

        elif self._type == "sf":
            for values in parameter_sets:
                prog = self.backend
                prog.reset()
                _apply_sf_layer(prog, values)  # type: ignore[arg-type]
                state = Statevector.from_instruction(prog)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)

        else:  # TorchQuantum
            for values in parameter_sets:
                self.backend(self.q_device, torch.tensor(values, dtype=torch.float32))
                row = [self.q_device.states.view(-1)[0].item()]  # placeholder
                results.append(row)

        if self.shots is None:
            return results

        # Add shot noise
        noisy: List[List[complex]] = []
        rng = np.random.default_rng()
        for row in results:
            noisy_row = [complex(rng.normal(val.real, 1 / self.shots)) + 1j * rng.normal(val.imag, 1 / self.shots) for val in row]
            noisy.append(noisy_row)
        return noisy

    # --------------------------------------------------------------------- #
    # Quantum kernel helper (TorchQuantum)
    # --------------------------------------------------------------------- #

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix using the underlying TorchQuantum kernel.
        """
        if self._type!= "tq":
            raise RuntimeError("Kernel matrix only available for TorchQuantum backend.")
        kernel = self.backend
        return np.array([[float(kernel(x, y)) for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Factory helpers – fraud detection (SF), classifier (Qiskit), kernel (TorchQuantum)
# --------------------------------------------------------------------------- #

# Fraud‑detection parameters (mirrors the seed)
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the fraud‑detection model."""
    if sf is None:
        raise RuntimeError("Strawberry Fields is not installed.")
    program = sf.Program(2)
    with program.context as q:
        _apply_sf_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_sf_layer(q, layer, clip=True)
    return program


def _apply_sf_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool = False,
) -> None:
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


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List, List, List[SparsePauliOp]]:
    """Build a simple layered ansatz with explicit encoding and variational parameters."""
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is not installed.")
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


# TorchQuantum kernel (seed)
if tq is not None:
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
        """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
        kernel = Kernel()
        return np.array([[float(kernel(x, y)) for y in b] for x in a])

else:
    Kernel = None  # type: ignore[assignment]
    kernel_matrix = None  # type: ignore[assignment]

__all__ = [
    "HybridEstimator",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "Kernel",
    "kernel_matrix",
]
