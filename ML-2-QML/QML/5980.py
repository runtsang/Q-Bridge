import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """
    Variational sampler QNN implemented with Pennylane.

    The circuit encodes two input angles via RY gates,
    applies a tunable number of layers of rotation‑and‑entanglement
    blocks, and samples the computational basis.
    The circuit parameters are stored as a flat NumPy array
    and updated during training with a gradient‑based optimiser.
    """

    def __init__(self,
                 num_qubits: int = 2,
                 depth: int = 2,
                 device_name: str = "default.qubit",
                 shots: int = 1024,
                 seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.device = qml.device(device_name, wires=num_qubits, shots=shots, seed=seed)

        # Parameter vector: 2 input RY angles + 4*depth rotation angles
        self.params = np.random.uniform(0, 2 * np.pi,
                                        size=(2 + 4 * depth))
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            # Encode inputs
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)

            idx = 2
            for _ in range(self.depth):
                # Rotations
                qml.RY(params[idx], wires=0)
                qml.RY(params[idx + 1], wires=1)
                idx += 2
                # Entangling layer
                qml.CNOT(0, 1)
                qml.RZ(params[idx], wires=0)
                qml.RZ(params[idx + 1], wires=1)
                idx += 2

            # Measure probabilities
            return qml.probs(wires=range(self.num_qubits))

        return circuit

    def sample(self, n_shots: int | None = None) -> np.ndarray:
        """
        Sample computational basis states from the circuit.

        Parameters
        ----------
        n_shots : int, optional
            Number of shots to draw. If None, uses the circuit's default shots.
        """
        shots = n_shots if n_shots is not None else self.shots
        probs = self.qnode(self.params)
        outcomes = np.random.choice(len(probs), size=shots, p=probs)
        hist = np.bincount(outcomes, minlength=len(probs)) / shots
        return hist

    def set_params(self, new_params: np.ndarray) -> None:
        """Replace the internal parameter vector."""
        self.params = new_params

    def get_params(self) -> np.ndarray:
        """Return the current parameter vector."""
        return self.params

__all__ = ["SamplerQNN"]
