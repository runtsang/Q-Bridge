import pennylane as qml
import numpy as np
from typing import List, Tuple

class QuantumClassifierModel:
    """Data‑re‑uploading variational classifier implemented with Pennylane."""

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[qml.QNode, List[qml.Symbol], List[qml.Symbol], List[qml.PauliOp]]:
        """
        Construct a variational circuit that repeatedly uploads the data
        and entangles adjacent qubits with a CNOT ladder.

        Parameters
        ----------
        num_qubits : int
            Number of qubits (also the dimensionality of the input vector).
        depth : int
            Number of data‑re‑uploading layers.

        Returns
        -------
        circuit : qml.QNode
            The PennyLane quantum node ready for gradient‑based training.
        encoding : List[qml.Symbol]
            Symbols representing the input features.
        weights : List[qml.Symbol]
            Symbols representing the trainable rotation angles.
        observables : List[qml.PauliOp]
            Pauli‑Z observables on each qubit for the output layer.
        """
        dev = qml.device("default.qubit", wires=num_qubits)

        # Symbols for input encoding and trainable parameters
        encoding = [qml.Symbol(f"x{i}") for i in range(num_qubits)]
        weights = [qml.Symbol(f"theta{i}") for i in range(num_qubits * depth)]

        @qml.qnode(dev)
        def circuit(x, theta):
            # Initial data encoding
            for i, qubit in enumerate(range(num_qubits)):
                qml.RX(x[i], wires=qubit)

            idx = 0
            for _ in range(depth):
                # Variational rotations
                for qubit in range(num_qubits):
                    qml.RY(theta[idx], wires=qubit)
                    idx += 1
                # Entangling CNOT ladder
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

                # Re‑upload data
                for i, qubit in enumerate(range(num_qubits)):
                    qml.RX(x[i], wires=qubit)

            # Expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

        observables = [qml.PauliZ(wires=i) for i in range(num_qubits)]
        return circuit, encoding, weights, observables
