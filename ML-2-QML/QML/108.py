import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode, Device
from typing import Tuple

class SamplerQNNGen:
    """
    Quantum sampler implemented with PennyLane.

    The circuit consists of:
    - Two qubits with independent rotation parameters (`theta`)
    - A configurable entangling block (CX or CZ)
    - Linear combination of rotation and entanglement parameters
    - Probability distribution over computational basis obtained by measurement

    Parameters
    ----------
    num_qubits : int
        Number of qubits (default 2)
    entangler : str
        Type of entanglement gate ('cx' or 'cz') (default 'cx')
    device_name : str
        PennyLane device backend (default 'default.qubit')
    seed : int
        Random seed for the device (default 42)
    """
    def __init__(
        self,
        num_qubits: int = 2,
        entangler: str = "cx",
        device_name: str = "default.qubit",
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.entangler = entangler
        self.dev = qml.device(device_name, wires=num_qubits, shots=1000, seed=seed)

        # Parameter shapes: one rotation per qubit, one entanglement per pair
        self.param_shapes = (num_qubits, 1)

        @qnode(self.dev, interface="autograd")
        def circuit(theta: np.ndarray, ent: np.ndarray) -> np.ndarray:
            # Rotations
            for i in range(num_qubits):
                qml.RY(theta[i], wires=i)
            # Entanglement
            if entangler == "cx":
                for i in range(num_qubits - 1):
                    qml.CX(wires=[i, i + 1])
            else:
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            # Additional rotations after entanglement
            for i in range(num_qubits):
                qml.RY(ent[i], wires=i)
            # Measure probabilities
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def sample(self, theta: np.ndarray, ent: np.ndarray) -> np.ndarray:
        """
        Return the probability distribution over the computational basis.
        Parameters
        ----------
        theta : np.ndarray
            Rotation parameters for each qubit.
        ent : np.ndarray
            Entanglement parameters for each qubit (post‑entanglement rotations).
        """
        probs = self.circuit(theta, ent)
        return probs

    def trainable_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return initial random parameters for training.
        """
        rng = np.random.default_rng()
        theta = rng.uniform(-np.pi, np.pi, size=self.num_qubits)
        ent = rng.uniform(-np.pi, np.pi, size=self.num_qubits)
        return theta, ent

__all__ = ["SamplerQNNGen"]
