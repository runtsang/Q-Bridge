"""Quantum sampler network based on PennyLane.

The class wraps a PennyLane QNode that prepares a parameterised state
using Ry rotations and CNOT entanglement.  It exposes a ``sample`` method
that returns a probability distribution over the computational basis
states.  The QNode is differentiable, enabling hybrid training with the
classical SamplerQNN defined above.
"""

import pennylane as qml
import numpy as np
from typing import Sequence

class SamplerQNN:
    """
    Quantum sampler using a variational circuit.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the sampler.
    depth : int, default 2
        Number of alternating layers of Ry rotations and CNOTs.
    device : str, optional
        Name of the PennyLane device; if None, a default statevector
        device is created.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        device: str | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_qubits) if device is None else qml.device(device, wires=num_qubits)

        # Parameters: input (one per qubit) + weight (depth * num_qubits)
        self.input_params = np.arange(num_qubits, dtype=np.float64)
        self.weight_params = np.arange(num_qubits * depth, dtype=np.float64)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float], weights: Sequence[float]) -> Sequence[float]:
            """Variational circuit preparing a state that depends on inputs and weights."""
            # Input encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for d in range(depth):
                for i in range(num_qubits):
                    qml.RY(weights[d * num_qubits + i], wires=i)
                # Entangle with a simple linear chain
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Return probability amplitudes
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def sample(self, inputs: Sequence[float], weights: Sequence[float]) -> np.ndarray:
        """Return a probability distribution over all basis states."""
        return self.circuit(inputs, weights)

    def gradient(self, inputs: Sequence[float], weights: Sequence[float]) -> np.ndarray:
        """Compute the gradient of the probability distribution w.r.t. weights."""
        return qml.grad(self.circuit)(inputs, weights)

__all__ = ["SamplerQNN"]
