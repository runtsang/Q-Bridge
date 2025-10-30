"""SamplerQNNGen312 – quantum variational sampler."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

__all__ = ["SamplerQNNGen312"]


class SamplerQNNGen312:
    """
    A 3‑qubit variational sampler built with Pennylane.
    The circuit consists of an input encoding layer, two entangling
    layers, and a set of parameterised rotations.  The sampler
    returns measurement counts for a specified number of shots.

    Parameters
    ----------
    device : pennylane.Device, optional
        Quantum device to execute the circuit on.  Defaults to
        ``qml.device("default.qubit", wires=3)``.
    shots : int, default 1024
        Number of measurement shots for sampling.
    seed : int, optional
        Random seed for reproducible parameter initialisation.
    """
    def __init__(self, device: qml.Device | None = None, shots: int = 1024, seed: int | None = None) -> None:
        self.device = device or qml.device("default.qubit", wires=3)
        self.shots = shots
        self.seed = seed
        rng = np.random.default_rng(seed)
        # 12 parameters: 3 for input encoding, 3 for entangling, 6 for rotations
        self.params = rng.uniform(0, 2 * np.pi, 12)

        @qml.qnode(self.device, interface="numpy")
        def circuit(params: np.ndarray) -> np.ndarray:
            # Input encoding: RX rotations
            for i in range(3):
                qml.RX(params[i], wires=i)
            # Entangling layers
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            # Parameterised rotations
            for i in range(3, 12):
                qml.RY(params[i], wires=(i - 3) % 3)
            # Return full probability distribution over all 3 qubits
            return qml.probs(wires=[0, 1, 2])

        self.circuit = circuit

    def sample(self, params: np.ndarray | None = None) -> dict[str, int]:
        """
        Generate measurement counts for the current circuit parameters.

        Parameters
        ----------
        params : np.ndarray, optional
            Parameter vector to use for sampling.  If None, the
            instance's stored parameters are used.

        Returns
        -------
        dict[str, int]
            Mapping from binary string outcomes to observed counts.
        """
        if params is None:
            params = self.params
        probs = self.circuit(params)
        # Draw samples according to the probability distribution
        outcomes = np.random.choice(len(probs), size=self.shots, p=probs)
        counts: dict[str, int] = {}
        for outcome in outcomes:
            key = format(outcome, f"0{len(probs)}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def probabilities(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Return the full probability vector for the given parameters.
        """
        if params is None:
            params = self.params
        return self.circuit(params)
