"""Quantum convolutional filter built on Pennylane.

The class implements a hardware‑efficient variational ansatz that
acts on a kernel‑sized patch.  The circuit is parameterised by a
set of rotation angles that can be trained on a real backend.
The `run` method accepts a 2‑D array and returns the average
probability of measuring |1⟩ across all qubits, matching the
behaviour of the original quantum filter.
"""

import numpy as np
import pennylane as qml

class ConvEnhancedQuantum:
    """
    Quantum convolutional filter using Pennylane.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float, default 0.5
        Threshold used to map classical pixel values to π or 0 rotation.
    shots : int, default 1024
        Number of shots for the simulator.
    device_name : str, default "default.qubit"
        Pennylane device name.
    layers : int, default 2
        Depth of the variational ansatz.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.5,
                 shots: int = 1024,
                 device_name: str = "default.qubit",
                 layers: int = 2):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.device = qml.device(device_name, wires=kernel_size**2, shots=shots)
        self.layers = layers
        # initialise parameters randomly
        self.params = np.random.uniform(0, 2 * np.pi,
                                        size=(layers, kernel_size**2))
        self._circuit = qml.QNode(self._variational_circuit, self.device)

    def _variational_circuit(self, params, data):
        # data is a 1‑D array of length n_qubits
        n = self.kernel_size**2
        # encode data as rotation angles
        for i in range(n):
            angle = np.pi if data[i] > self.threshold else 0.0
            qml.RY(angle, wires=i)
        # variational layers
        for l in range(self.layers):
            for i in range(n):
                qml.RY(params[l, i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
        # measure expectation of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    def run(self, data):
        """
        Evaluate the quantum filter on a 2‑D kernel.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits.
        """
        data_vec = np.reshape(data, (self.kernel_size**2,))
        # bind the data to the circuit
        expectations = self._circuit(self.params, data_vec)
        # convert expectation values from [-1,1] to probabilities of |1⟩
        probs = (1 - np.array(expectations)) / 2
        return probs.mean()

__all__ = ["ConvEnhancedQuantum"]
