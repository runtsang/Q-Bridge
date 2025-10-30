"""Hybrid quantum‑classical convolutional filter with auto‑search and noise support."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.providers import BackendV2
from typing import Iterable, Optional, Sequence

__all__ = ["ConvQuantumEnhanced"]


class ConvQuantumEnhanced:
    """
    Extends the original QuanvCircuit with:

    * Parameterised RX angles that can be tuned.
    * Optional noise‑model simulation.
    * Auto‑search over threshold and a single angle value.
    * A flexible run interface that accepts a 2‑D array.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: Optional[BackendV2] = None,
        shots: int = 200,
        threshold: int = 127,
        noise_model: Optional[qiskit.providers.MultiprocessingBackend] = None,
        angle: float = np.pi / 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.noise_model = noise_model
        self.angle = angle

        # Parameterised angles
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Base circuit
        self._circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray | Sequence[float]) -> float:
        """
        Execute the circuit on a single 2‑D data patch.

        Parameters
        ----------
        data : np.ndarray or sequence
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        arr = np.asarray(data).reshape(1, self.n_qubits)
        param_binds = []
        for dat in arr:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
            noise_model=self.noise_model,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

    def auto_search(
        self,
        dataset: Iterable[np.ndarray],
        angle_grid: Sequence[float] = np.linspace(0, np.pi, 5),
        threshold_grid: Sequence[int] = [0, 64, 127],
    ) -> Tuple[float, int]:
        """
        Grid search over angle and threshold values to maximise average output.

        Parameters
        ----------
        dataset : Iterable
            Iterable of 2‑D arrays to evaluate.
        angle_grid : Sequence[float]
            Candidate angle values.
        threshold_grid : Sequence[int]
            Candidate threshold values.

        Returns
        -------
        Tuple[float, int]
            Best angle and threshold found.
        """
        best_score = -float("inf")
        best_angle = self.angle
        best_thresh = self.threshold

        for ang in angle_grid:
            for thresh in threshold_grid:
                # Re‑build circuit with new angle
                self.angle = ang
                for i in range(self.n_qubits):
                    self._circuit.data[i].operation.params = [ang]
                self.threshold = thresh

                # Evaluate on dataset
                scores = [self.run(d) for d in dataset]
                avg = sum(scores) / len(scores)

                if avg > best_score:
                    best_score = avg
                    best_angle = ang
                    best_thresh = thresh

        # Update to best found parameters
        self.angle = best_angle
        self.threshold = best_thresh
        for i in range(self.n_qubits):
            self._circuit.data[i].operation.params = [best_angle]

        return best_angle, best_thresh
