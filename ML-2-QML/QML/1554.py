import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """
    Variational quantum circuit emulating a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int, default=1
        Number of qubits in the circuit.
    depth : int, default=1
        Number of variational layers.
    shots : int, default=1000
        Number of shots for expectation estimation.
    device : str or pennylane.Device, optional
        Pennylane device to use. If None, defaults to the simulator
        ``default.qubit``.
    """

    def __init__(self, n_qubits: int = 1,
                 depth: int = 1,
                 shots: int = 1000,
                 device: str | qml.Device | None = None):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.device = device or qml.device("default.qubit", wires=n_qubits, shots=shots)

        # Total number of parameters: n_qubits * depth
        self.n_params = n_qubits * depth

        # Define the QNode
        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            # params shape: (depth, n_qubits)
            for layer in range(depth):
                for q in range(n_qubits):
                    qml.RY(params[layer, q], wires=q)
                # Entangling layer – a simple CZ chain
                for q in range(n_qubits - 1):
                    qml.CZ(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameter vectors.

        Parameters
        ----------
        thetas : np.ndarray or list
            Iterable of shape (batch_size, n_params). Each row contains
            the flattened parameter vector for one circuit evaluation.

        Returns
        -------
        np.ndarray
            Array of shape (batch_size,) with the expectation values.
        """
        thetas = np.asarray(thetas, dtype=np.float32)
        if thetas.ndim!= 2 or thetas.shape[1]!= self.n_params:
            raise ValueError(f"thetas must have shape (batch, {self.n_params})")
        expectations = []
        for params in thetas:
            params_reshaped = params.reshape(self.depth, self.n_qubits)
            expectations.append(self.circuit(params_reshaped))
        return np.array(expectations)

    def gradient(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Compute gradients via the parameter‑shift rule for a batch of parameters.

        Parameters
        ----------
        thetas : np.ndarray or list
            Iterable of shape (batch_size, n_params).

        Returns
        -------
        np.ndarray
            Gradient array of shape (batch_size, n_params).
        """
        thetas = np.asarray(thetas, dtype=np.float32)
        if thetas.ndim!= 2 or thetas.shape[1]!= self.n_params:
            raise ValueError(f"thetas must have shape (batch, {self.n_params})")
        grads = []
        shift = np.pi / 2
        for params in thetas:
            params = params.reshape(self.depth, self.n_qubits)
            grad = np.zeros_like(params)
            for i in range(self.depth):
                for q in range(self.n_qubits):
                    idx = i * self.n_qubits + q
                    # Shift +π/2
                    params_plus = np.copy(params)
                    params_plus[i, q] += shift
                    val_plus = self.circuit(params_plus)
                    # Shift –π/2
                    params_minus = np.copy(params)
                    params_minus[i, q] -= shift
                    val_minus = self.circuit(params_minus)
                    grad[i, q] = 0.5 * (val_plus - val_minus)
            grads.append(grad.flatten())
        return np.array(grads)
