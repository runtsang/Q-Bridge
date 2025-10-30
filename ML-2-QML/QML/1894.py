import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Quantum incremental data‑uploading classifier with a second encoding layer
    and controlled‑phase entanglement.  The factory mirrors the classical
    interface, returning the circuit, encoding parameters, variational
    parameters, and measurement observables.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int,
                                 depth: int,
                                 **kwargs) -> Tuple[QuantumCircuit,
                                                    Iterable,
                                                    Iterable,
                                                    List[SparsePauliOp]]:
        """
        Construct a layered ansatz with two encoding layers and a controlled‑phase
        entangler to increase expressivity.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # First encoding layer: RX rotations
        for qubit in range(num_qubits):
            circuit.rx(encoding[qubit], qubit)

        # Second encoding layer: RY rotations
        for qubit in range(num_qubits):
            circuit.ry(encoding[qubit], qubit)

        # Variational layers with controlled‑phase entanglers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.rz(weights[idx], qubit)
                idx += 1
            # Controlled‑phase entangler
            for qubit in range(num_qubits - 1):
                circuit.cu1(np.pi / 4, qubit, qubit + 1)
            # Additional CZ entanglement in a ring topology
            for qubit in range(num_qubits):
                circuit.cz(qubit, (qubit + 1) % num_qubits)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables
