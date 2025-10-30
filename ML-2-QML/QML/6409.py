"""
ConvFraudHybrid: Quantum module that fuses a quantum convolution circuit with a photonic fraud detection program.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Quantum convolution
# ----------------------------------------------------------------------
class QuanvCircuit:
    """
    Implements a 2‑D quantum filter that encodes a kernel‑sized image into qubit rotations.
    The circuit returns the average probability of measuring |1> across all qubits.
    """
    def __init__(self, kernel_size: int, backend: qiskit.providers.BaseBackend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for vec in flat:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, vec)}
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total = 0
        for bitstring, freq in counts.items():
            total += sum(int(bit) for bit in bitstring) * freq
        return total / (self.shots * self.n_qubits)

# ----------------------------------------------------------------------
# Photonic fraud detection program
# ----------------------------------------------------------------------
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
    return max(-bound, min(value, bound))

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

def build_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# ----------------------------------------------------------------------
# Combined hybrid quantum model
# ----------------------------------------------------------------------
class ConvFraudHybrid:
    """
    Hybrid quantum generator that first encodes an image kernel into a qubit‑based convolution circuit,
    then feeds the resulting probability amplitude into a photonic fraud‑detection program.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        conv_shots: int = 1024,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv_circuit = QuanvCircuit(
            kernel_size=conv_kernel_size,
            backend=backend,
            shots=conv_shots,
            threshold=conv_threshold,
        )

        self.fraud_prog = build_photonic_program(
            fraud_input_params if fraud_input_params else FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            fraud_layers if fraud_layers is not None else [],
        )
        self.engine = sf.Engine("fock", backend_options={"cutoff_dim": 5})

    def run(self, data: np.ndarray) -> float:
        """
        Execute the full hybrid pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (conv_kernel_size, conv_kernel_size) representing an image patch.

        Returns
        -------
        float
            Fraud‑risk score derived from the photonic program’s measurement statistics.
        """
        conv_out = self.conv_circuit.run(data)

        # Build a new photonic program that injects conv_out as a displacement amplitude
        prog = sf.Program(2)
        with prog.context as q:
            # Encode conv_out into both modes
            Dgate(conv_out, 0) | q[0]
            Dgate(conv_out, 0) | q[1]
            # Copy the rest of the gates from the base program
            for gate, mode in self.fraud_prog.program[0].gates:
                gate | mode

        result = self.engine.run(prog)
        state = result.state

        # Compute a simple fraud‑risk metric: mean photon number in mode 0
        mean_photon = state.expectation_value(sf.ops.N(0))
        return mean_photon

__all__ = ["ConvFraudHybrid", "FraudLayerParameters", "build_photonic_program"]
