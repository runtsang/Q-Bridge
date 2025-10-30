"""Quantum implementation of the hybrid convolutional filter.

The class `ConvGen034` mirrors the classical counterpart but replaces
the 2‑D convolution with a parameterised variational circuit that
encodes each pixel as a rotation angle.  The circuit is built from
a Z‑feature map (as in QCNN) followed by a TwoLocal ansatz.  A
single‑qubit expectation value of Z is returned, emulating the
probability of measuring |1> in the original quanvolution.
The design incorporates pooling through a controlled‑not network
and a clipping strategy for the rotation angles, inspired by
FraudDetection.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from dataclasses import dataclass
from typing import Iterable


@dataclass
class QuantumConvParams:
    """Configuration for the quantum convolutional filter."""
    kernel_size: int = 3
    n_layers: int = 2
    reps: int = 1
    entanglement: str = "full"
    backend: qiskit.providers.Provider | None = None
    shots: int = 1024
    clip: bool = True
    clip_bound: float = 5.0


class ConvGen034:
    """Quantum convolutional filter with feature mapping and ansatz.

    The circuit encodes a kernel-sized image into rotation angles
    via a ZFeatureMap.  A TwoLocal ansatz applies trainable
    rotations and entanglement.  The expectation value of the first
    qubit's Z observable is used as the filter output.
    """

    def __init__(self, params: QuantumConvParams | None = None) -> None:
        if params is None:
            params = QuantumConvParams()
        self.params = params

        self.n_qubits = self.params.kernel_size ** 2
        self.feature_map = ZFeatureMap(self.n_qubits, reps=1)
        self.ansatz = TwoLocal(
            self.n_qubits,
            rotation_blocks="ry",
            entanglement=self.params.entanglement,
            reps=self.params.reps,
            entanglement_blocks="cz",
        )

        # Combine feature map and ansatz into a single circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.append(self.feature_map, range(self.n_qubits))
        self.circuit.append(self.ansatz, range(self.n_qubits))
        # Measure all qubits
        self.circuit.measure_all()

        self.backend = self.params.backend or Aer.get_backend("qasm_simulator")
        self.estimator = Estimator(
            backend=self.backend,
            shots=self.params.shots,
        )
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single kernel image.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size) with
                  pixel intensities in [0, 1].

        Returns:
            float: expectation value of the first qubit's Z observable.
        """
        flat = data.reshape(1, self.n_qubits)
        # Clip pixel values to avoid extreme angles
        if self.params.clip:
            flat = np.clip(flat, 0, self.params.clip_bound)

        # Bind feature map parameters
        param_binds = [{p: val for p, val in zip(self.feature_map.parameters, row)}
                       for row in flat]
        # Bind ansatz parameters (initialized randomly)
        ansatz_params = np.random.uniform(0, 2 * np.pi, self.ansatz.num_parameters)
        param_binds = [{p: val for p, val in zip(self.ansatz.parameters, ansatz_params)}
                       for _ in param_binds]

        # Execute the circuit
        result = self.estimator.run(
            circuits=[self.circuit],
            parameter_binds=param_binds,
            observables=[self.observable],
        )
        expectation = result[0]
        return expectation

def Conv() -> ConvGen034:
    """Factory returning a pre‑configured quantum convolutional filter."""
    return ConvGen034()

__all__ = ["ConvGen034", "Conv", "QuantumConvParams"]
