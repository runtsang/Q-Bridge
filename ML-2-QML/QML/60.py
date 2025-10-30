"""Quantum fully‑connected layer using Pennylane.

The circuit implements a depth‑controlled ansatz of Ry rotations on a
single qubit.  The class is wrapped in a Pennylane `QNode` so that the
expectation value of Pauli‑Z can be evaluated efficiently.  The
`run` method mimics the original API and returns a NumPy array.
"""

import pennylane as qml
import numpy as np
from typing import Iterable, List, Tuple


def FCL(
    n_qubits: int = 1,
    dev_name: str = "default.qubit",
    shots: int = 1024,
    depth: int = 2,
) -> qml.QNode:
    """Return a parameterised quantum circuit that can be used in a hybrid model.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the ansatz.
    dev_name:
        Pennylane device name.
    shots:
        Number of shots for the simulator.
    depth:
        Number of Ry layers.
    """

    dev = qml.device(dev_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(thetas: List[float]) -> qml.numpy.Tensor:
        """Parameterised ansatz."""
        # Global H to create superposition
        qml.Hadamard(wires=list(range(n_qubits)))
        # Layered Ry gates
        for d in range(depth):
            for w in range(n_qubits):
                qml.RY(thetas[d * n_qubits + w], wires=w)
        # Expectation of Pauli‑Z on first qubit
        return qml.expval(qml.PauliZ(0))

    def run(thetas: Iterable[float]) -> np.ndarray:
        """Convenience wrapper that returns a NumPy array."""
        import torch
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32)
        expval = circuit(theta_tensor).detach().numpy()
        return np.array([expval])

    # Expose the run method as an attribute for API compatibility
    circuit.run = run
    return circuit


__all__ = ["FCL"]
