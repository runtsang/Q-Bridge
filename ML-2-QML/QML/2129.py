"""Hybrid quantum convolutional filter.

The class mirrors the classical interface but delegates to a variational
quantum circuit.  The circuit encodes each pixel value into an RX rotation
angle and applies a shallow entangling layer.  The measurement of the
Pauli‑Z expectation on each qubit is averaged to produce a scalar output.
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from typing import Iterable
from qiskit.providers import BaseBackend
from qiskit.providers.aer import AerSimulator

__all__ = ["HybridConv"]


class HybridConv:
    """
    Quantum implementation of the Conv filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the kernel (must match the classical counterpart).
    threshold : float, default 0.0
        Threshold used to binarise pixel values before encoding.
    backend : qiskit.providers.BaseBackend, optional
        Backend used for execution.  If None a local qasm_simulator
        is constructed.
    shots : int, default 100
        Number of shots per circuit execution.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        backend: BaseBackend | None = None,
        shots: int = 100,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

        self.shots = shots
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Create the parameterised variational circuit."""
        qc = QuantumCircuit(self.n_qubits)
        # Encode data via RX rotations
        data_params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(data_params):
            qc.rx(p, i)

        # Add a shallow entangling layer (CX pairs)
        for i in range(0, self.n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, self.n_qubits - 1, 2):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.measure_all()
        self.data_params = data_params
        return qc

    def run(self, data: np.ndarray | Iterable[Iterable[float]]) -> float:
        """
        Execute the circuit on a single 2‑D kernel patch.

        Parameters
        ----------
        data : array‑like of shape (kernel_size, kernel_size)
            Pixel values in the range [0, 255] (uint8).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten and binarise data
        flat = np.asarray(data, dtype=np.float32).flatten()
        bind = {}
        for i, val in enumerate(flat):
            angle = np.pi if val > self.threshold else 0.0
            bind[self.data_params[i]] = angle

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute expectation value of Z (probability of |1>)
        total_ones = 0
        total_counts = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * count
            total_counts += count

        prob = total_ones / (self.shots * self.n_qubits)
        return prob

    def run_batch(self, data_batch: np.ndarray) -> np.ndarray:
        """
        Run the filter on a batch of kernel patches.

        Parameters
        ----------
        data_batch : np.ndarray, shape (B, kernel_size, kernel_size)

        Returns
        -------
        np.ndarray, shape (B,)
            Output per batch element.
        """
        outputs = []
        for patch in data_batch:
            outputs.append(self.run(patch))
        return np.array(outputs)
