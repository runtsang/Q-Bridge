"""Quantum counterpart of ConvHybrid using Qiskit.

The circuit mirrors the classical convolution by operating on a 2×2 pixel patch
represented by four qubits.  It incorporates a fraud‑detection inspired
parameter set to control single‑qubit rotations, and applies a small
entangling block before measurement.  The output is the average probability
of measuring |1> across all qubits, analogous to the classical mean
activation.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic‑style layer."""
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


class ConvHybrid:
    """Quantum filter that emulates a 2×2 convolution with fraud‑detection style parameters."""

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 0.5,
        fraud_params: FraudLayerParameters | None = None,
        fraud_clip: bool = True,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        if fraud_params is None:
            fraud_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        self.fraud_params = fraud_params
        self.fraud_clip = fraud_clip
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the parameterised quantum circuit."""
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Encode pixel values via RX rotations
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)

        # Fraud‑detection style rotations per qubit
        for i in range(self.n_qubits):
            # Beam‑splitter like rotation
            self.circuit.rx(
                _clip(self.fraud_params.bs_theta, 5.0) if self.fraud_clip else self.fraud_params.bs_theta,
                i,
            )
            # Phase shift
            self.circuit.rz(
                _clip(self.fraud_params.bs_phi, 5.0) if self.fraud_clip else self.fraud_params.bs_phi,
                i,
            )
            # Individual phase gates
            self.circuit.rz(
                _clip(self.fraud_params.phases[i % 2], 5.0) if self.fraud_clip else self.fraud_params.phases[i % 2],
                i,
            )
            # Squeezing‑like rotations (mapped to RZ)
            self.circuit.rz(
                _clip(self.fraud_params.squeeze_phi[i % 2], 5.0) if self.fraud_clip else self.fraud_params.squeeze_phi[i % 2],
                i,
            )
            # Displacement‑like rotations
            self.circuit.rz(
                _clip(self.fraud_params.displacement_phi[i % 2], 5.0) if self.fraud_clip else self.fraud_params.displacement_phi[i % 2],
                i,
            )
            # Kerr‑like small rotations
            self.circuit.rz(
                _clip(self.fraud_params.kerr[i % 2], 1.0) if self.fraud_clip else self.fraud_params.kerr[i % 2],
                i,
            )

        # Simple entangling block
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on a 2×2 pixel patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with integer pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flattened = np.reshape(data, (self.n_qubits,))
        bind = {}
        for i, val in enumerate(flattened):
            bind[self.theta[i]] = np.pi if val > self.threshold else 0.0

        job = qiskit.execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = sum(freq * sum(int(bit) for bit in bitstring) for bitstring, freq in counts.items())
        return total_ones / (self.shots * self.n_qubits)

    def update_fraud_params(self, new_params: FraudLayerParameters) -> None:
        """Replace the fraud‑detection parameters and rebuild the circuit."""
        self.fraud_params = new_params
        self._build_circuit()


__all__ = ["ConvHybrid"]
