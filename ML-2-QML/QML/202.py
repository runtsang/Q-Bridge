"""Quantum variational classifier with configurable entanglement and measurement basis."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             entanglement: str = "nearest",
                             measurement_basis: str = "Z",
                             encoding: str = "rx",
                             parameter_sharing: bool = False) -> Tuple[QuantumCircuit,
                                                                            Iterable[ParameterVector],
                                                                            Iterable[ParameterVector],
                                                                            List[SparsePauliOp]]:
    """
    Construct a flexible layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    entanglement : {"nearest", "full", "none"}, default="nearest"
        Entanglement scheme between qubits.
    measurement_basis : {"Z", "X", "Y"}, default="Z"
        Basis in which to measure each qubit.
    encoding : {"rx", "ry", "rz", "ryx"}, default="rx"
        Single‑qubit encoding gate applied to each qubit.
    parameter_sharing : bool, default=False
        Whether to share parameters across layers, reducing the total number of
        trainable parameters.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed variational circuit.
    encoding_params : Iterable[ParameterVector]
        Parameters used for data encoding.
    weight_params : Iterable[ParameterVector]
        Variational parameters.
    observables : List[SparsePauliOp]
        Pauli operators for measurement.
    """
    # Parameter vectors
    encoding_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("theta", num_qubits * depth if not parameter_sharing else num_qubits)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding_params):
        if encoding == "rx":
            circuit.rx(param, qubit)
        elif encoding == "ry":
            circuit.ry(param, qubit)
        elif encoding == "rz":
            circuit.rz(param, qubit)
        elif encoding == "ryx":
            circuit.ry(param, qubit)
            circuit.rx(param, qubit)
        else:
            raise ValueError(f"Unsupported encoding gate: {encoding}")

    # Variational layers
    idx = 0
    for layer in range(depth):
        # Apply single‑qubit rotations
        for qubit in range(num_qubits):
            theta = weight_params[qubit] if parameter_sharing else weight_params[idx]
            circuit.ry(theta, qubit)
            if not parameter_sharing:
                idx += 1

        # Entanglement
        if entanglement == "nearest":
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cz(i, j)
        elif entanglement == "none":
            pass
        else:
            raise ValueError(f"Unsupported entanglement scheme: {entanglement}")

    # Measurement observables
    if measurement_basis == "Z":
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    elif measurement_basis == "X":
        observables = [SparsePauliOp("I" * i + "X" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    elif measurement_basis == "Y":
        observables = [SparsePauliOp("I" * i + "Y" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    else:
        raise ValueError(f"Unsupported measurement basis: {measurement_basis}")

    return circuit, list(encoding_params), list(weight_params), observables

class QuantumClassifierModel:
    """Convenience wrapper around a variational ansatz for classification.

    The wrapper exposes the circuit, encoding and weight parameters, and the
    measurement observables, mirroring the interface of the classical
    ``QuantumClassifierModel`` class.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 entanglement: str = "nearest",
                 measurement_basis: str = "Z",
                 encoding: str = "rx",
                 parameter_sharing: bool = False) -> None:
        self.circuit, self.encoding_params, self.weight_params, self.observables = build_classifier_circuit(
            num_qubits,
            depth,
            entanglement=entanglement,
            measurement_basis=measurement_basis,
            encoding=encoding,
            parameter_sharing=parameter_sharing,
        )
        self.num_qubits = num_qubits
        self.depth = depth

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying variational circuit."""
        return self.circuit

    def get_parameters(self) -> Tuple[Iterable[ParameterVector], Iterable[ParameterVector]]:
        """Return the encoding and variational parameters."""
        return self.encoding_params, self.weight_params

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the list of measurement observables."""
        return self.observables

__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]
