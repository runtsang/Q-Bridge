"""Quantum convolutional filter using a parameter‑shared variational circuit."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ConvEnhancedQML:
    """
    Variational quantum circuit that emulates a convolutional kernel.
    The circuit is depth‑wise: each qubit receives a rotation that
    depends on the corresponding pixel value, followed by a shallow
    entangling layer. The result is a probability‑weighted activation
    map that can be used as a drop‑in replacement for Conv.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel. The circuit operates on
        kernel_size**2 qubits.
    dev : pennylane.Device, optional
        Quantum device to run the circuit on. If None, a default
        Aer simulator is used.
    threshold : float, default=0.5
        Threshold used to binarise the input before encoding.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        dev: qml.Device | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold

        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev)
        def circuit(pixel_vals: np.ndarray, params: np.ndarray) -> np.ndarray:
            for i in range(self.n_qubits):
                qml.RX(pixel_vals[i] * np.pi, wires=i)

            # Shared rotation parameters
            for i in range(self.n_qubits):
                qml.RZ(params[i], wires=i)

            # Shallow entangling layer (nearest‑neighbour)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        # Initialise parameters
        self.params = pnp.random.uniform(0, 2 * np.pi, self.n_qubits)

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        pixel_vals = data.reshape(self.n_qubits)
        # Binarise according to threshold
        pixel_vals = np.where(pixel_vals > self.threshold, 1.0, 0.0)
        expvals = self.circuit(pixel_vals, self.params)
        probs = (1 - np.array(expvals)) / 2  # Convert PauliZ expectation to |1> probability
        return probs.mean().item()
