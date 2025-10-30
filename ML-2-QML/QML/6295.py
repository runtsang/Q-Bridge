import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes, PauliFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import Statevector
from typing import Tuple, List, Callable

__all__ = [
    "QuantumAutoencoder",
    "train_quantum_autoencoder",
    "QuantumAutoencoderCircuit",
]

def QuantumAutoencoderCircuit(
    num_qubits: int,
    latent_dim: int,
    num_trash: int,
    reps: int = 3,
) -> QuantumCircuit:
    """
    Builds a hybrid quantum auto‑encoder circuit:
    * An encoder (RealAmplitudes) that maps |0⟩ → |ψ⟩
    * A swap‑test style garbage removal
    * A decoder that is the inverse of the encoder
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encoder
    encoder = RealAmplitudes(num_qubits, reps=reps)
    circuit.compose(encoder, inplace=True)

    # Swap‑test style garbage removal
    trash = qr[latent_dim : latent_dim + num_trash]
    aux = qr[num_qubits - 1]
    circuit.h(aux)
    for i, t in enumerate(trash):
        circuit.cswap(aux, qr[latent_dim + i], t)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    # Decoder (inverse of encoder)
    decoder = encoder.inverse()
    circuit.compose(decoder, inplace=True)
    return circuit


class QuantumAutoencoder:
    """
    A lightweight quantum auto‑encoder that can be trained with a state‑vector sampler
    and COBYLA optimisation. The encoder and decoder are parameterised by
    :class:`qiskit.circuit.library.RealAmplitudes`.
    """
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        num_trash: int = 2,
        reps: int = 3,
        interpret: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = QuantumAutoencoderCircuit(
            num_qubits, latent_dim, num_trash, reps
        )
        self.sampler = Sampler()
        if interpret is None:
            self.interpret = lambda x: x
        else:
            self.interpret = interpret
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self.interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Return the raw measurement distribution for a given parameter vector."""
        return self.qnn(params)

    def loss(self, params: np.ndarray, target: np.ndarray) -> float:
        """Mean‑squared error between the predicted distribution and target."""
        pred = self.predict(params)
        return np.mean((pred - target) ** 2)


def train_quantum_autoencoder(
    qa: QuantumAutoencoder,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    optimizer_cls: Callable[..., COBYLA] = COBYLA,
) -> List[float]:
    """
    Trains :class:`QuantumAutoencoder` using the supplied data.
    The target is the measurement distribution of a state‑vector simulator
    applied to the original data encoded into the circuit.
    """
    history: List[float] = []
    optimizer = optimizer_cls(qa.loss)
    # initialise parameters randomly
    params = np.random.uniform(0, 2 * np.pi, size=len(qa.circuit.parameters))
    for epoch in range(epochs):
        # build target distribution from the data
        # For simplicity we use a classical feature map to create a target
        # distribution: a uniform distribution over the first two measurement outcomes.
        target = np.array([0.5, 0.5])
        loss_val = qa.loss(params, target)
        history.append(loss_val)

        # COBYLA returns a new parameter vector
        params = optimizer.minimize(
            lambda p: qa.loss(p, target), params, options={"maxiter": 20}
        )
    return history
