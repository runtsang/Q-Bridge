import numpy as np
from typing import Optional, Callable, List
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, GradientSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build a quantum auto‑encoder circuit with a swap‑test."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(
        RealAmplitudes(num_latent + num_trash, reps=5),
        range(0, num_latent + num_trash),
        inplace=True,
    )
    circuit.barrier()
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit

class AutoencoderQNN(SamplerQNN):
    """Quantum neural network that implements a variational auto‑encoder."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        circuit = _auto_encoder_circuit(num_latent, num_trash)
        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )

class Autoencoder:
    """Convenience wrapper that exposes a classical API for the quantum auto‑encoder."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        self.qnn = AutoencoderQNN(num_latent, num_trash)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def decode(self, latents: np.ndarray) -> np.ndarray:
        probs = []
        for x in latents:
            out = self.qnn(x, params=None)
            probs.append(out[1])
        return np.array(probs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.decode(inputs)

def train_qautoencoder(
    autoencoder: Autoencoder,
    data: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 0.01,
    callback: Optional[Callable[[int, float], None]] = None,
) -> List[float]:
    """Train the quantum auto‑encoder with a simple gradient‑descent loop."""
    qnn = autoencoder.qnn
    param_values = np.array([p.value for p in qnn.parameters], dtype=float)
    sampler = qnn.sampler
    grad_sampler = GradientSampler(sampler, gradient_method="parameter-shift")
    history: List[float] = []

    target = 1.0

    for epoch in range(epochs):
        loss = 0.0
        grads = np.zeros_like(param_values)

        for x in data:
            out = qnn(x, params=param_values)
            prob = out[1]
            loss += (prob - target) ** 2

            def loss_fn(p):
                out_p = qnn(x, params=p)
                return (out_p[1] - target) ** 2

            grad = grad_sampler.gradient(loss_fn, param_values)
            grads += grad

        loss /= len(data)
        grads /= len(data)

        param_values -= lr * grads
        qnn.set_parameters(param_values)

        history.append(loss)
        if callback:
            callback(epoch, loss)

    return history

__all__ = [
    "Autoencoder",
    "AutoencoderQNN",
    "train_qautoencoder",
]
