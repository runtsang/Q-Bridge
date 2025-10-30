"""Hybrid quantum layer that combines a parameter‑tuned rotation grid with
entanglement and measurement statistics.

The implementation follows the same public interface as the classical
seed: a `run` method that takes a list of angles and a 2‑D data patch.
Each qubit in an `n_qubits = kernel_size**2` grid receives an
`rx(theta)` rotation, then a random two‑layer entangling circuit is
applied.  The expectation value is the average probability of
measuring |1> across all qubits.  The quantum backend is the
Aer QASM simulator.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit


class HybridQuantumLayer:
    """
    Quantum circuit that models a fully‑connected layer with a
    convolution‑style qubit grid.  Each qubit is rotated by a tunable
    angle derived from the `thetas` argument; the data patch controls
    the initial rotation via a threshold mapping.
    """

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127.0) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square data patch; determines qubit count.
        backend : qiskit.providers.BaseBackend, optional
            Quantum backend; defaults to Aer QASM simulator.
        shots : int
            Number of shots per circuit execution.
        threshold : float
            Value used to decide whether a data element triggers a
            π rotation in the initial data‑dependent layer.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Create parameterized circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta_params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Data‑dependent rotation layer
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta_params[i], i)

        self.circuit.barrier()

        # Entangling block mimicking a simple quanvolution filter
        self.circuit += random_circuit(self.n_qubits, depth=2)

        self.circuit.measure_all()

    def run(self, thetas: Iterable[float], data: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Execute the circuit on a single data patch.

        Parameters
        ----------
        thetas : Iterable[float]
            Tunable angles, one per qubit, that modulate the initial
            rotation layer.
        data : Sequence[Sequence[float]]
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            Average probability of measuring |1> across all qubits.
        """
        # Flatten data and map to π rotations based on threshold
        flat_data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for datum in flat_data:
            bind = {}
            for i, val in enumerate(datum):
                bind[self.theta_params[i]] = np.pi if val > self.threshold else 0.0
            # Overlay tunable angles
            bind.update({p: t for p, t in zip(self.theta_params, thetas)})
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute average |1> probability
        total_ones = 0
        for key, count in result.items():
            ones = sum(int(bit) for bit in key)
            total_ones += ones * count

        expectation = total_ones / (self.shots * self.n_qubits)
        return np.array([expectation])


def FCL() -> HybridQuantumLayer:
    """Return a quantum layer mimicking the original API."""
    return HybridQuantumLayer()


__all__ = ["FCL"]
