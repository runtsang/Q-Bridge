import pennylane as qml
import numpy as np
import torch
from pennylane import numpy as pnp

class SamplerQNN:
    """
    A hybrid quantum‑classical sampler.  The circuit consists of two qubits, a layer of
    input‑dependent rotations followed by an entangling block and a layer of trainable
    rotations.  The QNode returns the probability of measuring each computational basis
    state, which can be used directly as a probability vector or sampled from with
    torch.distributions.Categorical.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        entanglement: str = "cnot",
        weight_shape: tuple[int,...] = (4,),
        device: str = "default.qubit",
        seed: int | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.weight_shape = weight_shape
        self.dev = qml.device(device, wires=num_qubits)
        # initialise trainable weights
        self.weights = pnp.random.uniform(
            low=-np.pi, high=np.pi, size=weight_shape, requires_grad=True
        )

        # build the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # input‑dependent rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # entanglement
            if entanglement == "cnot":
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # trainable rotations
            for i in range(num_qubits):
                qml.RY(weights[i], wires=i)
            # measurement probabilities
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit and return a probability vector over 2^num_qubits states.
        Inputs must be a tensor of shape (..., num_qubits).
        """
        probs = self.circuit(inputs, self.weights)
        return probs

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the probability distribution defined by the circuit.
        The returned tensor has shape (..., num_samples) containing integer indices
        of the computational basis states.
        """
        probs = self.forward(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,)).permute(1, 0)

    def get_weights(self) -> torch.Tensor:
        """Return the current trainable weight tensor."""
        return self.weights

    def set_weights(self, new_weights: torch.Tensor) -> None:
        """Replace the internal weight tensor with ``new_weights``."""
        self.weights = new_weights

__all__ = ["SamplerQNN"]
