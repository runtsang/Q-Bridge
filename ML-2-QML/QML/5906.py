"""Quantum self‑attention implemented with Pennylane.

The circuit consists of a block of parameterised rotations (representing
the query and key projections) followed by a layer of entangling gates
(simulating the dot‑product).  The output measurement statistics are
interpreted as attention logits, which are then soft‑maxed and
applied to the value vector (classical).
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttentionModule:
    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        device: str = "default.qubit",
    ):
        """
        Parameters
        ----------
        num_qubits: int
            Number of qubits.  Must be at least 2.
        num_layers: int
            Depth of the variational circuit.
        device: str
            Pennylane device name.
        """
        assert num_qubits >= 2, "Need at least two qubits for entanglement"
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits)

    def _variational_block(self, params: pnp.ndarray):
        """Parameterised rotations followed by CNOT entanglement."""
        for i in range(self.num_qubits):
            qml.RY(params[3 * i], wires=i)
            qml.RZ(params[3 * i + 1], wires=i)
            qml.RX(params[3 * i + 2], wires=i)
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def _circuit(self, rotation_params: pnp.ndarray, entangle_params: pnp.ndarray, inputs: pnp.ndarray):
        """Build the full circuit returning measurement probabilities."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Rotation block
            self._variational_block(rotation_params)
            # Entanglement block
            self._variational_block(entangle_params)
            # Measure each qubit in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum attention circuit.

        Parameters
        ----------
        rotation_params: np.ndarray
            Shape (num_qubits * 3,) – rotation angles for the first layer.
        entangle_params: np.ndarray
            Shape (num_qubits * 3,) – rotation angles for the entanglement layer.
        inputs: np.ndarray
            Classical value vector to be weighted by the attention logits.
        shots: int
            Number of measurement shots for sampling.

        Returns
        -------
        output: np.ndarray
            Weighted sum of the value vector using the learned attention
            probabilities derived from the measurement statistics.
        """
        # Run the variational circuit
        probs = self._circuit(rotation_params, entangle_params, inputs)
        # Convert expectation values to probabilities in [0,1]
        probs = (np.array(probs) + 1) / 2
        probs = probs / probs.sum()  # softmax‑like normalisation

        # Classical attention application
        value = inputs.astype(float)
        output = probs[:, None] * value
        return output.sum(axis=0)

__all__ = ["SelfAttentionModule"]
