"""Quantum circuit builder for the HybridNATGraphQNN model.

The circuit is defined with PennyLane and can be differentiated with
PyTorch.  It uses a fixed adjacency graph to entangle the qubits.
The rotation parameters are shared across all samples and are
optimised jointly with the classical network.

The function returns a callable that accepts a feature vector
(`inputs`), a weight vector (`weights`), and the adjacency matrix
(`adjacency`) and produces a tensor of expectation values.
"""

from __future__ import annotations

import pennylane as qml
import torch

def create_quantum_circuit(n_wires: int, adjacency: torch.Tensor):
    """Create a PennyLane QNode that implements the quantum layer.

    Parameters
    ----------
    n_wires : int
        Number of qubits.
    adjacency : torch.Tensor
        2‑D tensor of shape (n_wires, n_wires) containing 0/1 entries
        that indicate which qubits should be entangled with CNOT gates.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A function that evaluates the circuit given a feature vector
        and a weight vector.
    """
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        """Quantum circuit for a single sample."""
        # Encode classical data with RX rotations
        for i in range(n_wires):
            qml.RX(inputs[i], wires=i)

        # Variational rotations
        for i in range(n_wires):
            qml.RZ(weights[i], wires=i)

        # Entanglement according to the adjacency matrix
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                if adjacency[i, j] == 1:
                    qml.CNOT(wires=[i, j])

        # Return expectation values of Z on every qubit
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(n_wires)])

    return circuit

__all__ = ["create_quantum_circuit"]
