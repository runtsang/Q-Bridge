"""HybridClassifierModel: quantum classifier with fast expectation evaluation and shot‑noise simulation."""
from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a variational ansatz for classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (feature dimension).
    depth : int
        Depth of the variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The parameterised circuit.
    encoding : list[ParameterVector]
        Parameters for data‑encoding rotations.
    weights : list[ParameterVector]
        Variational parameters.
    observables : list[SparsePauliOp]
        Z‑observables on each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data encoding (RX rotations)
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # Variational layers with full‑chain entanglement
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables

class HybridClassifierModel:
    """Quantum classifier with fast expectation evaluation and shot‑noise simulation."""
    def __init__(self, num_qubits: int, depth: int):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self._parameters = list(self.circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set.

        Parameters
        ----------
        parameter_sets : list of sequences
            Each inner sequence contains the values for all circuit parameters.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each observable.
        seed : int, optional
            Random seed for noise generation.

        Returns
        -------
        results : list[list[complex]]
            Expectation values per parameter set.
        """
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in self.observables]
            # Optional shot noise simulation
            if shots is not None:
                rng = np.random.default_rng(seed)
                noisy_row = [rng.normal(loc=val.real, scale=1/np.sqrt(shots)) for val in row]
                row = noisy_row
            results.append(row)
        return results

    def qasm(self) -> str:
        """Return the QASM representation of the underlying circuit."""
        return self.circuit.qasm()

__all__ = ["HybridClassifierModel"]
