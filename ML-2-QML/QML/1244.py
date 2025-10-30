import pennylane as qml
import numpy as np
from typing import Sequence

class SamplerQNN:
    """
    Variational quantum sampler built on Pennylane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (default 2).
    hidden_layers : int
        Number of entanglement layers.
    init_std : float
        Standard deviation for initializing trainable angles.
    device : str
        Pennylane device name (e.g., "default.qubit").

    Notes
    -----
    The circuit applies input‑dependent Ry rotations followed by a
    trainable variational block.  Sampling is performed by executing
    the circuit and measuring all qubits in the computational basis.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        hidden_layers: int = 2,
        init_std: float = 0.1,
        device: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.hidden_layers = hidden_layers
        self.dev = qml.device(device, wires=n_qubits)

        # Parameter shapes
        self.input_params = np.zeros(n_qubits)
        self.weight_params = np.random.normal(
            0.0, init_std, size=(hidden_layers, n_qubits)
        )

        # Build the qnode
        @qml.qnode(self.dev, interface="numpy")
        def circuit(inputs, weights):
            # Input‑dependent rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational block
            for layer, w in enumerate(weights):
                qml.RY(w[0], wires=0)
                qml.RY(w[1], wires=1)
                qml.CNOT(wires=[0, 1])
                if layer < hidden_layers - 1:
                    qml.CNOT(wires=[1, 0])
            # Measurement
            return qml.sample(wires=range(n_qubits))

        self.circuit = circuit

    def set_params(self, input_params: np.ndarray, weight_params: np.ndarray) -> None:
        """
        Update the circuit parameters.

        Parameters
        ----------
        input_params : array_like
            Input angles for Ry gates.
        weight_params : array_like
            Trainable angles per hidden layer.
        """
        self.input_params = np.asarray(input_params)
        self.weight_params = np.asarray(weight_params)

    def sample(self, n_shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return samples in the computational basis.

        Parameters
        ----------
        n_shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Array of shape (n_shots, n_qubits) with bitstrings.
        """
        raw_samples = self.circuit(self.input_params, self.weight_params)
        # Convert each bitstring to an integer label
        return np.array([int("".join(map(str, bits)), 2) for bits in raw_samples])

__all__ = ["SamplerQNN"]
