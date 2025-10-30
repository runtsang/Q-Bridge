"""Quantum sampler network with parameter‑shift gradient and re‑usable circuit."""

import pennylane as qml
import pennylane.numpy as np
from typing import Optional

class _SamplerQNN:
    """
    Variational quantum sampler on two qubits.
    Provides methods to evaluate probabilities, sample bitstrings,
    compute gradients via the parameter‑shift rule, and perform a simple
    gradient‑descent update.
    """

    def __init__(
        self,
        dev: qml.Device | None = None,
        num_params: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.dev = dev or qml.device("default.qubit", wires=2)
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.params = rng.randn(num_params)
        self.num_params = num_params

        @qml.qnode(self.dev, interface="torch")
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(params[2], wires=0)
            qml.RY(params[3], wires=1)
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def forward(self, params: np.ndarray) -> np.ndarray:
        """Return the probability vector for the supplied parameters."""
        return self.circuit(params)

    def sample(self, num_shots: int = 1024) -> np.ndarray:
        """Draw bitstring samples from the circuit."""
        probs = self.forward(self.params)
        indices = np.random.choice(len(probs), size=num_shots, p=probs)
        return np.array([np.binary_repr(i, width=2) for i in indices])

    def parameter_shift_grad(self) -> np.ndarray:
        """Compute gradient of the |00> probability w.r.t. each parameter."""
        grad = np.zeros_like(self.params)
        shift = np.pi / 2
        for i in range(self.num_params):
            plus = self.params.copy()
            minus = self.params.copy()
            plus[i] += shift
            minus[i] -= shift
            p_plus = self.circuit(plus)[0]
            p_minus = self.circuit(minus)[0]
            grad[i] = (p_plus - p_minus) / 2
        return grad

    def update(self, lr: float = 0.01) -> None:
        """Gradient‑descent update of the parameters."""
        grads = self.parameter_shift_grad()
        self.params -= lr * grads

def SamplerQNN() -> _SamplerQNN:
    """Factory returning a ready‑to‑use SamplerQNN instance."""
    return _SamplerQNN()
