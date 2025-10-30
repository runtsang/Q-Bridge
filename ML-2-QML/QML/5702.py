"""Quantum counterpart to the HybridLayer.

The quantum implementation follows the same interface as the classical version:
`run`, `weight_sizes`, and `observables`.  It constructs a parameterised ansatz that

  1. Encodes data via Rx rotations (`encoding` parameters).
  2. Applies a depth‑dependent sequence of Ry rotations (`weights`).
  3. Entangles neighboring qubits with CZ gates.
  4. Measures Z on each qubit and returns the weighted expectation value.

The class is designed to be a drop‑in replacement for the classical layer in hybrid
training loops or benchmark studies.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct the layered ansatz used by HybridLayer.

    Returns
    -------
    circuit : QuantumCircuit
        The full parameterised circuit.
    encoding : list[ParameterVector]
        Parameters used for the data‑encoding Rx gates.
    weights : list[ParameterVector]
        Variational parameters for Ry rotations.
    observables : list[SparsePauliOp]
        Z observables for each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class HybridLayer:
    """
    Quantum implementation of the hybrid layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Depth of the ansatz.
    backend : qiskit.providers.BaseBackend, optional
        Execution backend.  Defaults to the Aer qasm simulator.
    shots : int, default=100
        Number of shots for expectation estimation.
    """

    def __init__(self, num_qubits: int, depth: int, backend=None, shots: int = 100) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        (
            self.circuit,
            self.encoding_params,
            self.weights,
            self._observables,
        ) = _build_classifier_circuit(num_qubits, depth)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list containing first the encoding parameters followed by the variational weights.
        """
        if len(thetas)!= len(self.encoding_params) + len(self.weights):
            raise ValueError("Parameter list length does not match expected size.")

        param_bind = {
            **{p: theta for p, theta in zip(self.encoding_params, thetas[: len(self.encoding_params)])},
            **{p: theta for p, theta in zip(self.weights, thetas[len(self.encoding_params) :])},
        }

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self.circuit)

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def weight_sizes(self) -> List[int]:
        """Return the number of variational parameters per Ry block."""
        return [self.num_qubits] * self.depth

    def observables(self) -> List[SparsePauliOp]:
        """Return the list of Z observables used for measurement."""
        return self._observables


__all__ = ["HybridLayer"]
