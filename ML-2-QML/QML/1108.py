"""Quantum autoencoder built with PennyLane, featuring a swap‑test decoder and parameter‑shift training."""
import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    num_qubits: int
    latent_dim: int = 3
    num_trash: int = 2
    device: str = "default.qubit"
    shots: int = 1024
    optimizer: str = "Adam"
    max_iter: int = 200

    def __post_init__(self) -> None:
        if self.latent_dim <= 0 or self.num_trash < 0:
            raise ValueError("latent_dim must be >0 and num_trash >=0")


class AutoencoderModel:
    """Quantum autoencoder with swap‑test decoder and gradient‑shift training."""

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.device = qml.device(config.device, wires=config.num_qubits, shots=config.shots)
        self._build_circuit()

    def _build_circuit(self) -> None:
        latent = self.config.latent_dim
        trash = self.config.num_trash
        num_latent = latent
        num_trash = trash
        num_total = num_latent + 2 * num_trash + 1

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Encode input data as a feature vector
            qml.templates.AngleEmbedding(inputs, wires=range(num_latent + num_trash))
            # Parameterised ansatz for latent subspace
            qml.templates.RealAmplitudes(weights, wires=range(num_latent + num_trash), reps=1)
            # Swap‑test to compare latent with trash
            aux = num_latent + 2 * num_trash
            qml.Hadamard(wires=aux)
            for i in range(num_trash):
                qml.CSwap(wires=[aux, num_latent + i, num_latent + num_trash + i])
            qml.Hadamard(wires=aux)
            return qml.probs(wires=aux)

        self.circuit = circuit
        # Initialise parameters
        self.weights = np.random.uniform(0, 2 * np.pi, self.circuit.num_params)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Return the latent probability distribution for the given inputs."""
        return self.circuit(inputs, self.weights)

    def decode(self, inputs: np.ndarray) -> np.ndarray:
        """Decode by re‑applying the circuit; in this toy design we return the same output."""
        return self.circuit(inputs, self.weights)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.decode(inputs)

    def train(self, data: np.ndarray, *, lr: float = 0.01, verbose: bool = False) -> List[float]:
        """Train the circuit parameters to minimise reconstruction error."""
        opt = qml.AdamOptimizer(stepsize=lr) if self.config.optimizer.lower() == "adam" else qml.GradientDescentOptimizer(stepsize=lr)
        history: List[float] = []

        for epoch in range(self.config.max_iter):
            loss = 0.0
            for x in data:
                def loss_fn(w):
                    preds = self.circuit(x, w)
                    # target is a state with all probability on |0>
                    target = np.array([1.0, 0.0])
                    return np.mean((preds - target) ** 2)
                loss += loss_fn(self.weights)
                self.weights = opt.step(loss_fn, self.weights)
            loss /= len(data)
            history.append(loss)
            if verbose:
                print(f"Epoch {epoch+1:03d} – loss: {loss:.6f}")
        return history

    def evaluate(self, data: np.ndarray) -> float:
        """Return the average fidelity between target |0> and circuit output."""
        fidelities = []
        for x in data:
            probs = self.circuit(x, self.weights)
            fidelity = probs[0]  # probability of |0>
            fidelities.append(fidelity)
        return np.mean(fidelities)

    @classmethod
    def from_config(cls, config: AutoencoderConfig) -> "AutoencoderModel":
        return cls(config)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderModel",
]
