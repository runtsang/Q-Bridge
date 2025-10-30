"""Quantum convolution module.

This module defines a `Conv` class that implements a simple
variational quantum circuit used as a filter in a quanvolution
layer.  The circuit consists of a layer of RX gates whose
rotation angles are set by the input pixel values, followed
by a random 2‑layer circuit that entangles the qubits.  The
output is the average probability of measuring |1> across
all qubits.

Features
--------
* Batch inference – the ``run`` method accepts a batch of
  2‑D arrays.
* Learnable threshold – the class accepts a threshold that
  determines whether a pixel is encoded as a 0 or π rotation.
* Configurable shots – the number of samples per circuit
  execution.
* Uses Qiskit Aer simulator; no external hardware required.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer
from typing import Union

class Conv:
    """Quantum convolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the filter (must match the input image patch size).
    shots : int, default 100
        Number of shots for each circuit execution.
    threshold : float, default 0.5
        Pixel values above this threshold are encoded as a π rotation.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        # Build the circuit template.
        self.circuit_template = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit_template.rx(self.theta[i], i)
        self.circuit_template.barrier()
        self.circuit_template += random_circuit(self.n_qubits, 2)
        self.circuit_template.measure_all()

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

    def _prepare_bindings(self, data: np.ndarray) -> list[dict]:
        """Create a list of parameter bindings for each sample."""
        bindings = []
        for sample in data:
            flat = sample.flatten()
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(flat)}
            bindings.append(bind)
        return bindings

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on a batch of input patches.

        Parameters
        ----------
        data : np.ndarray
            3‑D array of shape ``(batch, kernel_size, kernel_size)``.
            If a 2‑D array is supplied, it is treated as a single
            sample.

        Returns
        -------
        float
            The average probability of measuring |1> across all qubits
            and all shots, normalised by the batch size.
        """
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim!= 3:
            raise ValueError("Input must be 2‑D or 3‑D array.")

        bindings = self._prepare_bindings(data)

        job = execute(
            self.circuit_template,
            self.backend,
            shots=self.shots,
            parameter_binds=bindings,
        )
        result = job.result()
        counts = result.get_counts(self.circuit_template)

        # Compute average probability of |1> over all qubits and shots.
        total_ones = 0
        total_shots = self.shots * self.n_qubits * len(data)
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq

        return total_ones / total_shots

__all__ = ["Conv"]
