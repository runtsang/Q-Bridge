import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane import qnode
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    num_latent: int = 3
    num_trash: int = 2
    reps: int = 5
    device: str = "default.qubit"
    shots: int = 1024

class Autoencoder__gen013:
    """Quantum autoencoder built with a RealAmplitudes ansatz and a SWAP test."""
    def __init__(self, config: AutoencoderConfig = AutoencoderConfig()) -> None:
        self.config = config
        self.num_qubits = config.num_latent + 2 * config.num_trash + 1
        self.dev = qml.device(config.device, wires=self.num_qubits, shots=config.shots)

        @qnode(self.dev)
        def _qnode(inputs: pnp.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
            # Feature encoding
            qml.templates.BasicEntanglerLayers(params=inputs, wires=range(self.num_qubits))
            # Variational circuit
            qml.templates.RealAmplitudes(weights, wires=range(self.num_qubits), reps=config.reps)
            # SWAP test for reconstruction fidelity
            aux = self.num_qubits - 1
            qml.Hadamard(wires=aux)
            for i in range(config.num_trash):
                qml.CSwap(wires=[aux, config.num_latent + i, config.num_latent + config.num_trash + i])
            qml.Hadamard(wires=aux)
            return qml.probs(wires=range(self.num_qubits))

        self._qnode = _qnode
        # Initialize weights
        self.weights = pnp.random.randn(self._qnode.num_params)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: returns the probability distribution over all qubits."""
        return self._qnode(x, self.weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.encode(x)

    def loss(self, x: np.ndarray) -> float:
        """Mean‑squared error between input amplitudes and circuit output."""
        y = self.encode(x)
        return np.mean((x - y) ** 2)

def train_autoencoder(
    model: Autoencoder__gen013,
    data: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    optimizer: Optional[qml.GradientDescentOptimizer] = None,
) -> List[float]:
    """Training loop for the quantum autoencoder."""
    if optimizer is None:
        optimizer = qml.GradientDescentOptimizer(lr)
    loss_history: List[float] = []

    for epoch in range(epochs):
        loss = model.loss(data)
        loss_history.append(loss)
        grads = optimizer.gradient(lambda w: model.loss(data), model.weights)
        model.weights = optimizer.apply_gradients(model.weights, grads)

    return loss_history

__all__ = ["Autoencoder__gen013", "AutoencoderConfig", "train_autoencoder"]
