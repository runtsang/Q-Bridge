"""HybridAttentionLayer – quantum implementation.

The quantum version reproduces the logical structure of the classical
HybridAttentionLayer using parameterised Qiskit circuits.  It returns
an expectation value that can be used directly as a feature for downstream
classical models or for gradient‑based optimisation via parameter‑shift.

Typical usage:

    from HybridAttentionLayer import HybridAttentionLayer
    layer = HybridAttentionLayer(n_qubits=4)
    output = layer.run(
        thetas=[0.1,  # linear rotation theta
                0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,  # rotations
                0.13, 0.14, 0.15, 0.16]  # entanglements
    )
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator


class HybridAttentionLayer:
    """Quantum hybrid layer combining a parameterised linear and self‑attention block."""

    def __init__(self,
                 n_qubits: int = 4,
                 backend: qiskit.providers.BaseBackend | None = None,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : list[float]
            Concatenated parameters:
            - first value: linear rotation angle (theta)
            - next ``3 * n_qubits`` values: rotation angles for each qubit
            - remaining ``n_qubits - 1`` values: entanglement angles

        Returns
        -------
        np.ndarray
            Expectation value of the final measurement, wrapped in a 1‑D array.
        """
        # Parse parameters
        theta = thetas[0]
        rot_vals = thetas[1:1 + 3 * self.n_qubits]
        ent_vals = thetas[1 + 3 * self.n_qubits:]

        # Build circuit
        circuit = QuantumCircuit(self.n_qubits)

        # Self‑attention style rotations
        for i in range(self.n_qubits):
            circuit.rx(rot_vals[3 * i], i)
            circuit.ry(rot_vals[3 * i + 1], i)
            circuit.rz(rot_vals[3 * i + 2], i)

        # Entangling CRX gates
        for i in range(self.n_qubits - 1):
            circuit.crx(ent_vals[i], i, i + 1)

        # Linear “output” rotation on qubit 0
        circuit.rx(theta, 0)

        # Measure all qubits
        circuit.measure_all()

        # Execute
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to expectation value
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)

        return np.array([expectation])


__all__ = ["HybridAttentionLayer"]
