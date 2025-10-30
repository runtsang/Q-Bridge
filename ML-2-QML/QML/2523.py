"""Hybrid quantum autoencoder that uses a variational circuit as encoder,
a swap‑test based decoder, and a SamplerQNN for latent sampling.
It follows the API of the classical counterpart so that the same
training loop can be reused with minimal changes.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_aer import AerSimulator

def _as_numpy(data: np.ndarray | list[float]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data, dtype=np.float32)
    return array.astype(np.float32)

class HybridAutoencoder:
    """Quantum autoencoder that embeds a RealAmplitudes ansatz as encoder,
    a swap‑test based decoder, and a SamplerQNN for latent sampling."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        trash_dim: int = 2,
        reps: int = 5,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps

        # Encoder ansatz
        self.encoder_ansatz = RealAmplitudes(
            num_qubits=latent_dim + trash_dim,
            reps=reps,
        )

        # Decoder swap‑test circuit
        self.decoder_circuit = self._build_decoder()

        # SamplerQNN for latent sampling
        self.sampler_qnn = SamplerQNN(
            circuit=self.decoder_circuit,
            input_params=[],
            weight_params=self.decoder_circuit.parameters,
            sampler=StatevectorSampler(),
        )

        # Simulation backend
        self.backend = AerSimulator(method="statevector")

    def _build_decoder(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent+trash
        qc.compose(self.encoder_ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)
        qc.barrier()

        # Swap test
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode classical data into a latent vector using the ansatz."""
        # Prepare input state as computational basis state
        qc = QuantumCircuit(self.latent_dim + self.trash_dim)
        for i, val in enumerate(inputs[: self.latent_dim + self.trash_dim]):
            qc.ry(val, i)
        qc.compose(self.encoder_ansatz, inplace=True)
        result = self.backend.run(qc).result()
        state = Statevector(result.get_statevector())
        # Return the amplitude of the first qubits as latent representation
        return np.abs(state.data[: self.latent_dim])

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent vector back to classical space via sampler."""
        # Map latents to parameters for the decoder circuit
        params = {p: val for p, val in zip(self.decoder_circuit.parameters, latents)}
        qc = self.decoder_circuit.bind_parameters(params)
        result = self.backend.run(qc).result()
        counts = result.get_counts()
        # Return measurement probabilities as reconstruction
        probs = np.array(list(counts.values()), dtype=np.float32)
        return probs

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        latents = self.encode(inputs)
        return self.decode(latents)

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 3,
    trash_dim: int = 2,
    reps: int = 5,
) -> HybridAutoencoder:
    return HybridAutoencoder(input_dim, latent_dim=latent_dim, trash_dim=trash_dim, reps=reps)

def train_autoencoder(
    model: HybridAutoencoder,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    optimizer_cls=COBYLA,
    device: str | None = None,
) -> list[float]:
    """Train the quantum autoencoder using a classical optimizer."""
    algorithm_globals.random_seed = 42
    opt = optimizer_cls()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs in data:
            inputs = _as_numpy(inputs)
            # Forward pass
            recon = model.forward(inputs)
            # Compute MSE loss (placeholder)
            loss = np.mean((recon - inputs) ** 2)
            epoch_loss += loss
            # Backward pass via optimizer
            opt.minimize(
                lambda params: np.mean(
                    (model.decode(params) - inputs) ** 2
                ),
                x0=np.random.rand(len(list(model.decoder_circuit.parameters))),
                jac=None,
            )
        epoch_loss /= len(data)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_autoencoder",
]
