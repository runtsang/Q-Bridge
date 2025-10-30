"""Hybrid quantum implementation of a fully‑connected quantum‑style layer
and classifier.

The circuit is composed of an initial fully‑connected sub‑circuit (H + Ry)
followed by the layered ansatz used in the ``QuantumClassifierModel`` seed.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridQuantumClassifier:
    """
    Quantum circuit that emulates a fully‑connected layer followed by a
    classifier ansatz. The API mirrors the classical counterpart: ``run``
    accepts a list of parameters and returns the expectation value of the
    observables.
    """

    def __init__(
        self,
        num_qubits: int = 1,
        depth: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """
        Construct the layered ansatz with an initial fully‑connected sub‑circuit.
        """
        # Parameters
        self.fc_params = ParameterVector("theta_fc", self.num_qubits)
        self.enc_params = ParameterVector("x", self.num_qubits)
        self.weight_params = ParameterVector("theta", self.num_qubits * self.depth)

        self.circuit = QuantumCircuit(self.num_qubits)

        # Fully‑connected sub‑circuit: H + Ry(theta) per qubit
        self.circuit.h(range(self.num_qubits))
        for qubit, param in zip(range(self.num_qubits), self.fc_params):
            self.circuit.ry(param, qubit)

        # Encoding
        for qubit, param in zip(range(self.num_qubits), self.enc_params):
            self.circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                self.circuit.ry(self.weight_params[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the provided parameters.  ``thetas`` must
        contain ``num_qubits`` + ``num_qubits`` + ``num_qubits*depth`` values
        in that order (fc, encoding, variational).
        """
        # Split parameters
        fc_vals = list(thetas[: self.num_qubits])
        enc_vals = list(thetas[self.num_qubits : 2 * self.num_qubits])
        weight_vals = list(thetas[2 * self.num_qubits :])

        # Bind parameters
        bindings = {
            **{p: v for p, v in zip(self.fc_params, fc_vals)},
            **{p: v for p, v in zip(self.enc_params, enc_vals)},
            **{p: v for p, v in zip(self.weight_params, weight_vals)},
        }

        bound_circuit = self.circuit.bind_parameters(bindings)

        job = execute(bound_circuit, backend=self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Convert counts to probabilities
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(state, 2) for state in counts.keys()])

        # Expectation value of each observable
        expectation = np.zeros(len(self.observables))
        for state, prob in zip(states, probs):
            bits = format(state, f"0{self.num_qubits}b")
            for idx, _ in enumerate(self.observables):
                expectation[idx] += ((-1) ** int(bits[idx])) * prob

        return expectation

    def expectation(self, thetas: Iterable[float]) -> np.ndarray:
        """Alias to ``run`` for clarity."""
        return self.run(thetas)


def FCL() -> HybridQuantumClassifier:
    """
    Factory compatible with the original ``FCL`` seed.  Returns an instance
    of the hybrid quantum classifier with default parameters.
    """
    return HybridQuantumClassifier()


__all__ = ["HybridQuantumClassifier", "FCL"]
