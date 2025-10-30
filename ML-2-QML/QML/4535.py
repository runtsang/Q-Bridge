"""Unified hybrid classifier – quantum counterpart.

The quantum implementation mirrors the classical architecture:
* Encoding parameters ``x`` are applied via RX gates.
* A depth‑wise variational ansatz uses Ry rotations and CZ entanglement.
* The circuit measures all qubits and computes expectation values of Z on each qubit.
* The resulting expectation vector has the same dimensionality as the classical sampler head.

Both sides expose a ``run`` method that accepts a flat list of parameters and returns a NumPy array of observables.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import Aer


class UnifiedHybridClassifier:
    """Quantum variational circuit with encoding and depth‑wise ansatz."""

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vectors
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)
        # Encoding
        for qubit in range(num_qubits):
            self.circuit.rx(self.encoding[qubit], qubit)

        # Variational layers
        w_idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[w_idx], qubit)
                w_idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Measurement of all qubits
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of length ``num_qubits + num_qubits * depth`` containing
            encoding parameters followed by variational weights.

        Returns
        -------
        np.ndarray
            Expectation values of Z on each qubit.
        """
        thetas = list(thetas)
        enc_vals = thetas[: self.num_qubits]
        weight_vals = thetas[self.num_qubits :]

        param_bind = [{self.encoding[i]: enc_vals[i]} for i in range(self.num_qubits)]
        param_bind += [{self.weights[i]: weight_vals[i]} for i in range(len(weight_vals))]

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = {state: cnt / self.shots for state, cnt in counts.items()}

        expectations = []
        for i in range(self.num_qubits):
            exp = 0.0
            for state, p in probs.items():
                # Qiskit measurement order is reversed: last qubit is first bit
                bit = int(state[self.num_qubits - 1 - i])
                exp += ((-1) ** bit) * p
            expectations.append(exp)
        return np.array(expectations)

    def get_parameter_dict(self) -> dict:
        """Return a mapping of parameter names to Parameter objects."""
        return {"encoding": self.encoding, "weights": self.weights}

    def freeze_encoder(self) -> None:
        """Placeholder for API compatibility; quantum encoder cannot be frozen."""
        pass


__all__ = ["UnifiedHybridClassifier"]
