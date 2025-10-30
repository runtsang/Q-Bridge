"""QuantumClassifierModel: quantum counterpart that constructs a parameterised
ansatz circuit.  It supports data‑re‑uploading, a modular entanglement
pattern and optional layer‑wise re‑uploading.  The interface is identical
to the classical implementation so that downstream training pipelines can
switch between classical and quantum models without code changes.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

__all__ = ["QuantumClassifierModel"]


class QuantumClassifierModel:
    """Factory for a parameterised quantum classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (also the dimensionality of the input feature vector).
    depth : int
        Number of variational layers.
    reupload : bool, optional
        Whether to re‑upload the data after each variational layer
        (data‑re‑uploading scheme).  Default is ``True``.
    entanglement : str, optional
        Entanglement pattern for the variational block.
        Options: ``'full'``, ``'cnot'`` (CZ between neighbours) or ``'none'``.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        *,
        reupload: bool = True,
        entanglement: str = "cnot",
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a simple layered ansatz with explicit encoding and
        variational parameters.

        Returns
        -------
        circuit : QuantumCircuit
            Parameterised circuit ready for simulation or execution.
        encoding : List[ParameterVector]
            Parameters used for data encoding.
        weights : List[ParameterVector]
            Variational parameters for each layer.
        observables : List[SparsePauliOp]
            Measurement operators for each qubit.
        """
        # Data encoding
        encoding = ParameterVector("x", num_qubits)

        # Variational parameters
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Initial encoding
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

        weight_idx = 0
        for layer in range(depth):
            # Variational rotation block
            for q in range(num_qubits):
                circuit.ry(weights[weight_idx], q)
                weight_idx += 1

            # Entangling pattern
            if entanglement == "full":
                for q in range(num_qubits):
                    for r in range(q + 1, num_qubits):
                        circuit.cz(q, r)
            elif entanglement == "cnot":
                for q in range(num_qubits - 1):
                    circuit.cz(q, q + 1)

            # Optional data re‑uploading
            if reupload:
                for q, param in enumerate(encoding):
                    circuit.rx(param, q)

        # Measurement operators
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, [encoding], [weights], observables
