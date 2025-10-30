"""Hybrid quantum classifier with optional sampler.

The class builds a layered ansatz that mirrors the classical
`build_classifier_circuit`.  Data encoding is performed by
parameterized RX gates, followed by a stack of variational
RY and CZ layers.  An optional quantum sampler can be attached
using Qiskit Machine Learning's `SamplerQNN`.

The API is intentionally compatible with the legacy
`QuantumClassifierModel` module: `build_classifier_circuit`
returns the circuit, the encoding parameters, the variational
parameters and a list of observables (Z‑Pauli on each qubit).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridClassifierSampler:
    """
    Quantum classifier that optionally wraps a parameterized sampler.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    use_sampler : bool, default=False
        Whether to attach a quantum sampler to the model.
    """

    def __init__(self, num_qubits: int, depth: int, use_sampler: bool = False) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_sampler = use_sampler

        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._make_circuit()

        if use_sampler:
            self.sampler = self._make_sampler()
        else:
            self.sampler = None

    def _make_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def _make_sampler(self):
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=self.circuit,
            input_params=self.encoding,
            weight_params=self.weights,
            sampler=sampler,
        )

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the attached sampler on a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, num_qubits) containing the values
            for the encoding parameters.

        Returns
        -------
        np.ndarray
            Sample probabilities for each input.

        Raises
        ------
        RuntimeError
            If the sampler is not enabled.
        """
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled for this instance.")
        return self.sampler.sample(inputs)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a quantum circuit and metadata compatible with the
        legacy quantum helper interface.

        Returns
        -------
        circuit : QuantumCircuit
            The constructed ansatz.
        encoding : Iterable[ParameterVector]
            Parameter vectors for data encoding.
        weights : Iterable[ParameterVector]
            Parameter vectors for variational layers.
        observables : List[SparsePauliOp]
            Z‑Pauli on each qubit as measurement observables.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        qc = QuantumCircuit(num_qubits)

        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables


__all__ = ["HybridClassifierSampler"]
