"""SamplerQNN: a variational quantum sampler implemented with PennyLane.

The circuit operates on two qubits and consists of a parameterised
rotation layer followed by a controlled‑phase entangling block.
The `sample` method returns samples from the probability distribution
obtained by measuring in the computational basis. This implementation
provides a direct quantum analogue of the classical sampler.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class SamplerQNN:
    """
    Variational quantum sampler.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    num_layers : int, default 2
        Number of variational layers.
    device : str or pennylane.Device, default 'default.qubit'
        Quantum device to execute the circuit.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        num_layers: int = 2,
        device: str | qml.Device = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits)

        # Trainable parameters: rotations for each layer and qubit
        self.params = pnp.random.uniform(0, 2 * np.pi, (num_layers, num_qubits, 3))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            # Encode classical inputs as Ry rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for layer in range(num_layers):
                for q in range(num_qubits):
                    qml.RY(params[layer, q, 0], wires=q)
                    qml.RZ(params[layer, q, 1], wires=q)
                    qml.RX(params[layer, q, 2], wires=q)
                # Entanglement
                for q in range(num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                # Wrap around entanglement
                qml.CNOT(wires=[num_qubits - 1, 0])

            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution for a single input vector.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (num_qubits,) containing classical features.

        Returns
        -------
        np.ndarray
            Probability vector of length 2 ** num_qubits.
        """
        return self.circuit(inputs, self.params)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (num_qubits,) containing classical features.
        num_samples : int, default 1
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of shape (num_samples, num_qubits) with bitstrings.
        """
        probs = self.forward(inputs)
        cum_probs = np.cumsum(probs)
        samples = []
        for _ in range(num_samples):
            r = np.random.rand()
            idx = np.searchsorted(cum_probs, r)
            bitstring = format(idx, f"0{self.num_qubits}b")
            samples.append([int(b) for b in bitstring])
        return np.array(samples)

__all__ = ["SamplerQNN"]
