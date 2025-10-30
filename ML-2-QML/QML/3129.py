"""Hybrid fraud detection using a quantum convolutional encoder and a photonic circuit.

The module defines a `FraudDetectionHybrid` class that first runs a
quantum convolutional circuit (`QuanvCircuit`) to encode the input data
into a probability value.  This value is then used to set the displacement
parameters of a photonic circuit built from `FraudLayerParameters`.  The
resulting program is executed on a Strawberry Fields Fock backend and
returns the final state vector, allowing further classical post‑processing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 1.  Parameters – identical to the original seed
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
# 2.  Quantum convolutional filter
# --------------------------------------------------------------------------- #

class QuanvCircuit:
    """Quantum filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# 3.  Photonic circuit construction
# --------------------------------------------------------------------------- #

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
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
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# --------------------------------------------------------------------------- #
# 4.  Hybrid model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid:
    """
    Hybrid quantum‑classical fraud detection model.

    The `run` method first executes a quantum convolutional circuit on the
    input data, then uses the resulting probability to set the displacement
    parameters of a photonic circuit.  The photonic program is executed on a
    Strawberry Fields Fock backend and the resulting state vector is returned.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        conv_kernel_size: int = 2,
        conv_threshold: float = 127,
        shots: int = 100,
    ) -> None:
        self.input_params = input_params
        self.layer_params = layer_params
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel_size, backend, shots, conv_threshold)

    def run(self, data: np.ndarray) -> sf.backends.FockBackend.State:
        """
        Run the hybrid model on a single 2‑D data sample.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        sf.backends.FockBackend.State
            The photonic state after execution.
        """
        conv_output = self.conv.run(data)  # scalar between 0 and 1

        # Update displacement parameters in both input and hidden layers
        new_input = replace(self.input_params, displacement_r=(conv_output, conv_output))
        new_layers = [
            replace(l, displacement_r=(conv_output, conv_output)) for l in self.layer_params
        ]

        program = build_fraud_detection_program(new_input, new_layers)

        backend = sf.backends.FockBackend(2)
        result = backend.run(program)
        return result.state

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QuanvCircuit",
    "FraudDetectionHybrid",
]
