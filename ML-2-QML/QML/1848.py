"""Quantum convolutional filter.

The Conv class implements a variational circuit that emulates a convolution
filter on a kernel‑sized patch.  It follows the original seed but adds a
few refinements: configurable noise model, parameter‑driven rotations,
and a clear separation between the circuit construction and execution.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from typing import Optional

class Conv:
    """
    Quantum convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    backend : str | Backend | None, default "qasm_simulator"
        Qiskit backend to execute the circuit.
    shots : int, default 1000
        Number of shots for measurement.
    threshold : float, default 127
        Threshold value used to encode the classical input into rotation angles.
    noise_model : Optional[qiskit.providers.models.NoiseModel]
        Noise model to attach to the backend.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: str | Backend | None = None,
        shots: int = 1000,
        threshold: float = 127,
        noise_model: Optional[qiskit.providers.models.NoiseModel] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")

        if noise_model is not None:
            self.backend = self.backend.with_noise_model(noise_model)

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        """Construct a parameterised variational circuit."""
        qc = qiskit.QuantumCircuit(self.n_qubits)
        params = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Encode the input as RX rotations
        for i, p in enumerate(params):
            qc.rx(p, i)

        # Add a simple entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.measure_all()
        self.params = params
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a kernel‑sized patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Normalised probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1)
        param_binds = []

        for val in flat:
            bind = {p: np.pi if val > self.threshold else 0.0 for p in self.params}
            param_binds.append(bind)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute average number of |1> outcomes
        total_ones = 0
        for outcome, count in result.items():
            total_ones += outcome.count("1") * count

        return total_ones / (self.shots * self.n_qubits)

__all__ = ["Conv"]
