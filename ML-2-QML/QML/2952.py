"""Combined quantum implementation of a fully‑connected layer.

The class builds a Qiskit circuit that mirrors the classical network:
  - An encoding layer of RX gates parameterised by input features.
  - A variational ansatz consisting of alternating RY rotations and CZ entanglers.
  - Observables are Pauli‑Z on each qubit, matching the classical output indices.

The `run` method accepts a list of parameters and returns the expectation
value of the observables with respect to the prepared state.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class FCL:
    """
    Quantum fully‑connected layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int, default 1
        Number of variational layers.
    shots : int, default 100
        Number of shots for simulation.
    backend : Optional[Backend], default Aer qasm_simulator
        Backend on which to execute the circuit.
    """

    def __init__(
        self,
        num_qubits: int = 1,
        depth: int = 1,
        shots: int = 100,
        backend=None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameters
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(self.encoding, range(num_qubits)):
            self.circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Observables: Pauli‑Z on each qubit
        self.observables: List[SparsePauliOp] = [
            SparsePauliOp.from_label("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        # Metadata identical to classical side
        self.encoding_indices: List[int] = list(range(num_qubits))
        self.weight_sizes: List[int] = [1] * (num_qubits * depth)
        self.observables_indices: List[int] = list(range(num_qubits))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit and return expectation values.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of parameters (first num_qubits are encoding,
            remaining are variational weights).

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the expectation value.
        """
        # Bind parameters
        param_binds = [
            {self.encoding[i]: thetas[i] for i in range(self.num_qubits)}
        ]
        var_params = thetas[self.num_qubits :]
        for idx, val in enumerate(var_params):
            param_binds[0][self.weights[idx]] = val

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_counts = sum(counts.values())
        expectation = 0.0
        for state, cnt in counts.items():
            prob = cnt / total_counts
            # Convert binary string to integer (reverse bits because Qiskit)
            value = int(state[::-1], 2)
            expectation += value * prob

        return np.array([expectation])

    def parameters_flatten(self) -> List[float]:
        """Return all circuit parameters as a flat list."""
        return list(self.encoding) + list(self.weights)


def FCL_factory(num_qubits: int = 1, depth: int = 1, shots: int = 100) -> FCL:
    """Convenience factory mirroring the original function interface."""
    return FCL(num_qubits, depth, shots)


__all__ = ["FCL", "FCL_factory"]
