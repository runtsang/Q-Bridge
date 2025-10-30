"""Quantum circuit factory for a configurable variational classifier.

The builder returns a circuit together with the parameter vectors used
for encoding and for the variational layers, as well as a list of Pauli
observables that are measured to obtain the quantum feature vector.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    encoding_gate: str = "rx",
    variational_gate: str = "ry",
    entanglement: str = "cz",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz with configurable encoding,
    variational gate and entanglement pattern.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    encoding_gate : str, optional
        Gate used for data encoding.  Must be a single‑qubit rotation
        gate supported by Qiskit (e.g., "rx", "ry", "rz").
    variational_gate : str, optional
        Gate used in the variational layers.
    entanglement : str, optional
        Entanglement pattern: "cz", "cx" or "none".

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised quantum circuit.
    encoding_params : list[Parameter]
        Parameters used for data encoding.
    variational_params : list[Parameter]
        Parameters used in the variational layers.
    observables : list[SparsePauliOp]
        Pauli Z observables on each qubit.
    """
    if encoding_gate not in {"rx", "ry", "rz"}:
        raise ValueError(f"Unsupported encoding_gate: {encoding_gate}")
    if variational_gate not in {"rx", "ry", "rz"}:
        raise ValueError(f"Unsupported variational_gate: {variational_gate}")

    encoding_params = ParameterVector("x", num_qubits)
    variational_params = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding_params):
        getattr(circuit, encoding_gate)(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        # Apply variational gate on each qubit
        for qubit in range(num_qubits):
            getattr(circuit, variational_gate)(variational_params[idx], qubit)
            idx += 1
        # Entanglement
        if entanglement == "cz":
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        elif entanglement == "cx":
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        elif entanglement == "none":
            pass
        else:
            raise ValueError(f"Unsupported entanglement pattern: {entanglement}")

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding_params), list(variational_params), observables
