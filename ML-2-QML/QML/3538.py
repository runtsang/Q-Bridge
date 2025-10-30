"""Hybrid quantum convolution with photonic‑inspired parameterised layers.

The returned object mimics the API of the original quanvolution filter:
``Conv() -> object`` with a ``.run(data)`` method.  The circuit first
applies a quantum convolution (random circuit) and then a stack of
parameterised layers that mirror the structure of the FraudDetection
photonic program.  Each layer applies per‑qubit rotations followed by
CNOT entanglement, providing a richer functional space than the
baseline QuanvCircuit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable, Sequence

# Parameters that describe one photonic‑style layer in the quantum circuit.
# Each field is a float list with length equal to the number of qubits.
class FraudLayerParameters:
    bs_theta: Sequence[float]  # rotation angles for Rx
    bs_phi: Sequence[float]    # rotation angles for Rz
    phases: Sequence[float]
    squeeze_r: Sequence[float]
    squeeze_phi: Sequence[float]
    displacement_r: Sequence[float]
    displacement_phi: Sequence[float]
    kerr: Sequence[float]


class QuanvCircuit:
    """Quantum hybrid convolution + fraud‑detection style layers."""

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        threshold: float = 0.0,
        layers_params: Iterable[FraudLayerParameters] | None = None,
    ):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.layers_params = layers_params or []

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Parameterised rotations per qubit.
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        # Random convolution layer.
        self._circuit += random_circuit(self.n_qubits, 2)
        # Fraud‑detection style parameterised layers.
        self._apply_fraud_layers()
        self._circuit.measure_all()

    def _apply_fraud_layers(self) -> None:
        """Append a stack of photonic‑style layers to the circuit."""
        for layer in self.layers_params:
            # Per‑qubit rotations.
            for i, (theta, phi) in enumerate(zip(layer.bs_theta, layer.bs_phi)):
                self._circuit.rx(theta, i)
                self._circuit.rz(phi, i)
            # Entanglement: CNOTs between neighbouring qubits.
            for i in range(0, self.n_qubits - 1, 2):
                self._circuit.cx(i, i + 1)

    def run(self, data) -> float:
        """Execute the circuit on classical data and return the mean |1> prob."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {
                self.theta[i]: np.pi if val > self.threshold else 0
                for i, val in enumerate(dat)
            }
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute average probability of measuring |1> across all qubits.
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


def Conv(
    kernel_size: int = 2,
    threshold: float = 0.0,
    layers_params: Iterable[FraudLayerParameters] | None = None,
    shots: int = 100,
    backend=None,
) -> QuanvCircuit:
    """Factory that returns a hybrid quantum convolution circuit."""
    return QuanvCircuit(
        kernel_size=kernel_size,
        threshold=threshold,
        layers_params=layers_params,
        shots=shots,
        backend=backend,
    )


__all__ = ["Conv", "FraudLayerParameters"]
