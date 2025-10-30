import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a data‑re‑uploading ansatz with parameter‑shift friendly outputs."""
    # Data encoding parameters
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters for each layer
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Initial data encoding
    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    # Layered variational block
    w_index = 0
    for _ in range(depth):
        # Single‑qubit rotations
        for q in range(num_qubits):
            circuit.ry(weights[w_index], q)
            w_index += 1
        # Entanglement: nearest‑neighbour CZ gates
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)
        # Re‑upload data
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables

__all__ = ["build_classifier_circuit"]
