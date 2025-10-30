import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
import torch

class SamplerQNN:
    """
    Hybrid quantum‑classical sampler based on a parameterised variational circuit.
    The circuit is defined with a repeated pattern of RY rotations followed by
    a fully‑connected CNOT grid.  Parameters are split into input and weight
    sets; the input set is fed into the first layer of rotations and is
    treated as a classical conditioning signal.
    """
    def __init__(self, num_qubits: int = 2,
                 entanglement: str = "full",
                 depth: int = 3,
                 device_name: str = "default.qubit",
                 device_shots: int = 1024):
        self.num_qubits = num_qubits
        self.entanglement = entanglement
        self.depth = depth
        self.dev = qml.device(device_name, wires=num_qubits, shots=device_shots)

        # Parameter shapes
        self.input_params = pnp.array(pnp.random.uniform(0, 2*np.pi,
                                                        (num_qubits,), requires_grad=False))
        self.weight_params = pnp.array(pnp.random.uniform(0, 2*np.pi,
                                                          (depth, num_qubits), requires_grad=True))

        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _entangle(self, wires):
        for i, w in enumerate(wires):
            target = wires[(i+1) % len(wires)]
            qml.CNOT(w, target)

    def _circuit(self, *params):
        # Split params into input and weight slices
        inp = params[0]  # shape (num_qubits,)
        weights = params[1:]  # list of depth arrays
        qml.RY(inp[0], 0)
        if self.num_qubits > 1:
            qml.RY(inp[1], 1)
        self._entangle(range(self.num_qubits))

        for w in weights:
            for q, theta in enumerate(w):
                qml.RY(theta, q)
            self._entangle(range(self.num_qubits))

        return qml.probs(wires=range(self.num_qubits))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability distribution over basis states.
        `inputs` is an array of shape (..., num_qubits) with values in [0, 2π].
        """
        probs = self.qnode(inputs, *self.weight_params)
        return torch.tensor(probs, dtype=torch.float32)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the quantum device using the current parameters.
        """
        probs = self.forward(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

    def loss(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Negative log‑likelihood of the target given the quantum output.
        """
        probs = self.forward(inputs)
        return -torch.log(probs.gather(-1, target.unsqueeze(-1))).mean()

__all__ = ["SamplerQNN"]
