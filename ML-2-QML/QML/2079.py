import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EnhancedSamplerQNN:
    """
    A hybrid quantum-classical sampler using a Pennylane variational circuit.
    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits in the circuit.
    entanglement : str or List[Tuple[int, int]], default 'full'
        Entanglement scheme.
    n_layers : int, default 2
        Number of variational layers.
    seed : int, optional
        Random seed for weight initialization.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 entanglement: str | list[tuple[int, int]] = 'full',
                 n_layers: int = 2,
                 seed: int | None = None):
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        self.n_layers = n_layers
        self.seed = seed

        # Parameter vector for rotation angles
        self.params = pnp.random.randn(n_layers * 3 * n_qubits) if seed is None else pnp.random.RandomState(seed).randn(n_layers * 3 * n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Input encoding via Ry rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[idx], wires=i); idx += 1
                    qml.RZ(weights[idx], wires=i); idx += 1
                    qml.RX(weights[idx], wires=i); idx += 1
                # Entanglement
                if entanglement == 'full':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif isinstance(entanglement, list):
                    for pair in entanglement:
                        qml.CNOT(wires=pair)
            # Measure probabilities of computational basis
            probs = qml.probs(wires=range(n_qubits))
            return probs

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute output probability distribution for a single input vector.
        Parameters
        ----------
        inputs : array-like, shape (n_qubits,)
            Classical input encoded into rotation angles.
        Returns
        -------
        probs : ndarray
            Probability distribution over 2^n_qubits outcomes.
        """
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape!= (self.n_qubits,):
            raise ValueError(f"Expected input shape ({self.n_qubits},), got {inputs.shape}")
        return self.circuit(inputs, self.params)

    def sample(self, inputs: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the probability distribution.
        Parameters
        ----------
        inputs : array-like, shape (n_qubits,)
            Classical input.
        n_samples : int, default 1
            Number of samples to draw.
        Returns
        -------
        samples : ndarray, shape (n_samples, n_qubits)
            Sampled bitstrings.
        """
        probs = self.__call__(inputs)
        return np.random.choice(len(probs), size=n_samples, p=probs)

__all__ = ["EnhancedSamplerQNN"]
