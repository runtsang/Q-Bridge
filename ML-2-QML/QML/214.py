"""Quantum circuit builder with parameter‑shift gradient support and measurement budget."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def _parameter_shift(circuit: QuantumCircuit, param: ParameterVector) -> List[float]:
    """Placeholder for a parameter‑shift gradient estimator. Returns a gradient list."""
    # In a real implementation this would evaluate the circuit at shifted
    # parameter values and compute the derivative. Here we return zeros.
    return [0.0 for _ in param]

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    measurement_budget: int | None = None,
    ansatz_type: str = "data_uploading",
) -> Tuple[
    QuantumCircuit,
    Iterable[ParameterVector],
    Iterable[ParameterVector],
    List[SparsePauliOp],
]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    The ``measurement_budget`` controls the number of shots used in simulation.
    The ``ansatz_type`` can be extended to support different quantum programs.
    """
    if ansatz_type not in {"data_uploading", "parametric"}:
        raise ValueError(f"Unsupported ansatz_type: {ansatz_type}")

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

    if measurement_budget is not None:
        circuit.metadata["shots"] = measurement_budget

    return circuit, list(encoding), list(weights), observables
