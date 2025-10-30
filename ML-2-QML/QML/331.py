"""Quantum‑enhanced convolutional filter.

This module implements :class:`ConvEnhanced`, a variational circuit that
mirrors the behaviour of the classical filter but operates on quantum
hardware or a simulator.  The circuit contains a trainable rotation
parameter for each qubit, a fixed entangling layer, and a measurement
of the probability of observing |1>.  The design allows the filter
to be trained end‑to‑end with a gradient‑based optimiser via
parameter‑shift rules (not implemented here but the interface
supports it).

The class is intentionally lightweight; it can be used as a drop‑in
replacement for the original `Conv` function in the codebase.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter


class ConvEnhanced:
    """
    Variational quantum filter with a learnable rotation for each qubit.
    The filter accepts a 2‑D array of shape ``(kernel_size, kernel_size)``
    and returns the average probability of measuring |1> across all qubits.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the classical kernel (also the linear size of the qubit grid).
    shots : int, default 100
        Number of shots for a qasm simulation (unused with state‑vector backend).
    threshold : float, default 127
        Pixel intensity threshold used to encode classical data into
        rotation angles.
    backend : qiskit.providers.backend.Backend, optional
        Quantum backend; defaults to the state‑vector simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 127,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("statevector_simulator")

        # Trainable rotation parameters
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Build the parameterised circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised single‑qubit rotations
        for i, theta in enumerate(self.theta):
            qc.rx(theta, i)
        # Simple nearest‑neighbour entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        return qc

    def set_parameters(self, weights: list | np.ndarray) -> None:
        """
        Set the rotation angles of the circuit.

        Parameters
        ----------
        weights : array‑like
            Length must equal ``n_qubits``.  Values are interpreted as
            rotation angles in radians.
        """
        if len(weights)!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} weights, got {len(weights)}"
            )
        bind = {theta: float(w) for theta, w in zip(self.theta, weights)}
        self.circuit.assign_parameters(bind, inplace=True)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on classical data and return the average
        probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : array‑like
            Shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            Average probability of |1> across all qubits.
        """
        arr = np.asarray(data, dtype=np.float32).reshape(self.n_qubits)
        param_binds = []
        for i, val in enumerate(arr):
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0}
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            parameter_binds=param_binds,
            shots=self.shots,
        )
        result = job.result()

        # For the state‑vector backend we can compute exact probabilities.
        if self.backend.name() == "statevector_simulator":
            statevector = result.get_statevector(self.circuit)
            probs = np.abs(statevector) ** 2
            probs_one = np.zeros(self.n_qubits)
            for idx, amp in enumerate(statevector):
                bits = np.binary_repr(idx, width=self.n_qubits)
                for i, bit in enumerate(bits):
                    if bit == "1":
                        probs_one[i] += np.abs(amp) ** 2
            return probs_one.mean()

        # Fallback to measurement counts for qasm simulators
        counts = result.get_counts(self.circuit)
        total_shots = sum(counts.values())
        probs_one = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            for i, bit in enumerate(reversed(state)):
                if bit == "1":
                    probs_one[i] += cnt
        return probs_one.mean() / total_shots

    def sample(self, data: np.ndarray, n_samples: int = 10) -> list:
        """
        Return a list of sampled bit‑strings for the given data.

        Parameters
        ----------
        data : array‑like
            Shape ``(kernel_size, kernel_size)``.
        n_samples : int, default 10
            Number of samples to draw.

        Returns
        -------
        list
            List of bit‑string samples.
        """
        arr = np.asarray(data, dtype=np.float32).reshape(self.n_qubits)
        param_binds = []
        for i, val in enumerate(arr):
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0}
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            parameter_binds=param_binds,
            shots=n_samples,
        )
        result = job.result()
        return result.get_counts(self.circuit).keys()

__all__ = ["ConvEnhanced"]
