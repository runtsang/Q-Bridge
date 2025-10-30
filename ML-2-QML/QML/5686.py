"""
QuantumNATGraphHybrid: Quantum circuit for the hybrid model.
"""

import pennylane as qml
import numpy as np

class QuantumCircuit:
    """Pennylane QNode that maps a classical vector to a quantum state
    via a parameterized circuit with a random layer."""
    def __init__(self, n_qubits: int = 4, seed: int | None = None):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        if seed is not None:
            np.random.seed(seed)
        # Random parameters for the circuit
        self.params = np.random.randn(n_qubits, 3)  # RX, RY, RZ per qubit
        self.random_layer = self._build_random_layer()

    def _build_random_layer(self):
        """Generate a random unitary layer as a list of gates."""
        ops = []
        for _ in range(10):
            qubit = np.random.randint(self.n_qubits)
            ops.append((qml.RX, np.random.randn(), qubit))
            ops.append((qml.CNOT, qubit, (qubit + 1) % self.n_qubits))
        return ops

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Apply the quantum circuit to the input features and return the
        state vector."""
        # Normalize input to [0, 1] and embed into the first qubit
        x = features / (np.linalg.norm(features) + 1e-12)
        return self._run_circuit(x)

    def _run_circuit(self, x: np.ndarray) -> np.ndarray:
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Encode classical input into the first qubit using RX
            qml.RX(x[0], wires=0)
            # Apply random layer
            for op, param, wire in self.random_layer:
                op(param, wires=wire)
            # Apply parameterized rotations
            for i in range(self.n_qubits):
                qml.RX(self.params[i, 0], wires=i)
                qml.RY(self.params[i, 1], wires=i)
                qml.RZ(self.params[i, 2], wires=i)
            return qml.state()
        return circuit()
