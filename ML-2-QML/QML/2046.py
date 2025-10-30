"""Quantum‑parameterized convolutional filter using Qiskit.

The `QuantumConvolution` class implements a variational circuit that
- applies a rotation around the X‑axis to each qubit;
- after a measurement, produces a feature‑map value between 0 and 1.
The class is intended to be used as a quantum back‑end for the `ConvGen` layer.

The implementation is lightweight, uses Qiskit’s Aer simulator by
default, and exposes a `get_feature_map` method that accepts a batch of
flattened patches and returns a tensor of feature values.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator


class QuantumConvolution:
    """Variational circuit used as a quantum convolutional filter."""

    def __init__(
        self,
        n_qubits: int,
        shots: int = 100,
        backend=None,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.threshold = 0.5  # threshold used to decide rotation angle

        # Build a reusable circuit template
        self._circuit_template = QuantumCircuit(self.n_qubits)
        self.theta = [
            Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i, th in enumerate(self.theta):
            self._circuit_template.rx(th, i)
        self._circuit_template.barrier()
        # Add a small random entangling layer for expressivity
        from qiskit.circuit.library import TwoLocal
        entangler = TwoLocal(
            num_qubits=self.n_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            repetitions=2,
            entanglement="circular",
        )
        self._circuit_template += entangler
        self._circuit_template.measure_all()

        # Pre‑compile the circuit
        self._compiled = transpile(
            self._circuit_template, backend=self.backend
        )

    def get_feature_map(self, patches: np.ndarray) -> np.ndarray:
        """Compute the quantum feature map for a batch of flattened patches.

        Parameters
        ----------
        patches : np.ndarray
            Shape (batch, n_qubits).  Each element is a real value that is
            thresholded to decide the rotation angle.

        Returns
        -------
        np.ndarray
            Feature values in [0, 1] of shape (batch, 1).
        """
        if patches.ndim!= 2 or patches.shape[1]!= self.n_qubits:
            raise ValueError(
                "patches must have shape (batch, n_qubits)."
            )

        param_binds = []
        for patch in patches:
            bind = {
                th: np.pi if val > self.threshold else 0
                for th, val in zip(self.theta, patch)
            }
            param_binds.append(bind)

        bound_circuits = [self._compiled.bind_parameters(b) for b in param_binds]
        qobj = assemble(bound_circuits, shots=self.shots, memory=False)
        results = self.backend.run(qobj).result()

        # Compute average probability of measuring |1> per qubit
        features = []
        for res in results:
            counts = res.get_counts()
            ones_sum = 0
            for bitstring, c in counts.items():
                ones = bitstring.count("1")
                ones_sum += ones * c
            prob = ones_sum / (self.shots * self.n_qubits)
            features.append([prob])

        return np.array(features)

__all__ = ["QuantumConvolution"]
