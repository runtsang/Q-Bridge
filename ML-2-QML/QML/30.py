"""Quantum sampler network using Pennylane variational circuit."""
import pennylane as qml
import pennylane.numpy as pnp
from pennylane import qnn

class SamplerQNN:
    """A two‑qubit variational sampler with entanglement and parameter‑shift training."""
    def __init__(self, dev=None, shots=1024) -> None:
        self.dev = dev or qml.device("default.qubit", wires=2, shots=shots)
        # Weight parameters for the variational circuit
        self.weight_params = pnp.random.uniform(0, 2 * pnp.pi, (4,))

        def circuit(inputs, weights):
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.sample(qml.PauliZ(0))

        # Build the Sampler QNN
        self.sampler_qnn = qnn.SamplerQNN(
            circuit=circuit,
            input_params=[qml.Symbol("x0"), qml.Symbol("x1")],
            weight_params=[qml.Symbol(f"w{i}") for i in range(4)],
            device=self.dev,
        )

    def forward(self, inputs: pnp.ndarray) -> pnp.ndarray:
        """Return the probability distribution over measurement outcomes."""
        return self.sampler_qnn(inputs, self.weight_params)

    def sample(self, inputs: pnp.ndarray, n_samples: int = 1) -> pnp.ndarray:
        """Draw samples from the quantum sampler."""
        probs = self.forward(inputs)
        # The sampler returns a probability vector for outcomes 0 and 1
        return pnp.random.choice([0, 1], size=n_samples, p=probs)

__all__ = ["SamplerQNN"]
