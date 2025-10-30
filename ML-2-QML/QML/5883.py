import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

class Autoencoder__gen285:
    """Quantum autoencoder using a variational circuit and swap‑test reconstruction.

    The circuit encodes input data into a latent subspace, then decodes it back.
    The training objective is to maximize fidelity between the input state and the
    decoded state using a swap‑test measurement. Noise models can be added for
    benchmarking.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 num_qubits: int | None = None,
                 ansatz_layers: int = 3,
                 device: str = "default.qubit",
                 shots: int = 1024):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits or (latent_dim + input_dim)
        self.device = device
        self.shots = shots

        self.dev = qml.device(device, wires=self.num_qubits, shots=shots)

        # Parameters for the ansatz
        self.params = pnp.random.uniform(0, 2*np.pi, (ansatz_layers, self.latent_dim))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            # Encode input data as rotations on the first input_dim qubits
            for i in range(self.input_dim):
                qml.RX(inputs[i], wires=i)

            # Encode latent subspace
            for layer, layer_params in enumerate(params):
                for j in range(self.latent_dim):
                    qml.RY(layer_params[j], wires=self.input_dim + j)

            # Decoder: reverse the encoding
            for layer, layer_params in reversed(list(enumerate(params))):
                for j in range(self.latent_dim):
                    qml.RY(-layer_params[j], wires=self.input_dim + j)

            # Swap‑test to estimate fidelity
            # Prepare ancilla qubit for swap test
            qml.Hadamard(wires=self.num_qubits - 1)
            for i in range(self.num_qubits - 1):
                qml.CSwap(wires=[self.num_qubits - 1, i, i])
            return qml.expval(qml.PauliZ(self.num_qubits - 1))

        self.circuit = circuit

    def fidelity(self, inputs: np.ndarray) -> float:
        return self.circuit(inputs, self.params)

    def loss(self, inputs: np.ndarray) -> float:
        # Loss is 1 - fidelity
        return 1 - self.fidelity(inputs)

    def train(self,
              data: np.ndarray,
              *,
              epochs: int = 100,
              lr: float = 0.01):
        opt = qml.AdamOptimizer(stepsize=lr)
        params = self.params

        for epoch in range(epochs):
            loss = 0.0
            for x in data:
                loss += self.loss(x)
            loss /= len(data)
            params = opt.step(self.loss, params)
            self.params = params

__all__ = ["Autoencoder__gen285"]
