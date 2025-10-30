"""Quantum‑based autoencoder built with Qiskit and a SamplerQNN decoder."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
#  CONFIGURATION
# --------------------------------------------------------------------------- #
class QuantumAutoEncoderConfig:
    """Configuration for the quantum autoencoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        trash_dim: int = 2,
        backend: str | None = None,
        shots: int = 1024,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.backend = backend
        self.shots = shots
        self.seed = seed

# --------------------------------------------------------------------------- #
#  QUANTUM ENCODER
# --------------------------------------------------------------------------- #
def _build_encoder_circuit(cfg: QuantumAutoEncoderConfig) -> QuantumCircuit:
    """Return a Qiskit circuit that encodes a classical vector into a latent state."""
    qr = QuantumRegister(cfg.latent_dim + 2 * cfg.trash_dim + 1, "q")
    cr = ClassicalRegister(1, "c")
    circ = QuantumCircuit(qr, cr)

    # Feature mapping via a parameterised ansatz
    circ.compose(
        RealAmplitudes(cfg.latent_dim + cfg.trash_dim, reps=5),
        range(0, cfg.latent_dim + cfg.trash_dim),
        inplace=True,
    )
    circ.barrier()

    # Swap‑test style entanglement with an ancilla
    ancilla = cfg.latent_dim + 2 * cfg.trash_dim
    circ.h(ancilla)
    for i in range(cfg.trash_dim):
        circ.cswap(ancilla, cfg.latent_dim + i, cfg.latent_dim + cfg.trash_dim + i)
    circ.h(ancilla)
    circ.measure(ancilla, cr[0])
    return circ

# --------------------------------------------------------------------------- #
#  QUANTUM DECODER (SamplerQNN)
# --------------------------------------------------------------------------- #
def _build_decoder_qnn(cfg: QuantumAutoEncoderConfig) -> SamplerQNN:
    """Build a SamplerQNN that decodes the latent state back to a classical vector."""
    # The circuit that will be parameterised by the QNN
    qc = QuantumCircuit(cfg.latent_dim)
    # The QNN will have no trainable parameters in this simple example;
    # we use a dummy circuit to satisfy the API.
    dummy_circuit = qc.copy()
    qnn = SamplerQNN(
        circuit=dummy_circuit,
        input_params=[],
        weight_params=[],
        interpret=lambda x: x,  # identity interpretation
        output_shape=(cfg.input_dim,),
        sampler=QiskitSampler(backend=cfg.backend, shots=cfg.shots, seed_simulator=cfg.seed),
    )
    return qnn

# --------------------------------------------------------------------------- #
#  QUANTUM AUTOENCODER
# --------------------------------------------------------------------------- #
class QuantumAutoEncoder:
    """Quantum autoencoder that maps a classical vector to a latent state and back."""
    def __init__(self, cfg: QuantumAutoEncoderConfig):
        self.cfg = cfg
        self.encoder_circuit = _build_encoder_circuit(cfg)
        self.decoder_qnn = _build_decoder_qnn(cfg)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a classical vector into a latent state via a quantum circuit."""
        # Prepare a circuit that encodes the input vector
        circ = self.encoder_circuit.copy()
        # Note: In a real implementation we would parameterise the circuit
        # with the input data; here we simply use the circuit as is.
        result = qiskit.execute(circ, backend=self.cfg.backend, shots=self.cfg.shots).result()
        counts = result.get_counts()
        # Convert measurement results to a probability vector
        probs = np.array([counts.get(bit, 0) for bit in sorted(counts)]) / self.cfg.shots
        return probs

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode a latent state back to a classical vector via a sampler."""
        # Use the SamplerQNN to obtain a probability distribution over the input space
        qnn_input = np.array([latent])  # batch dimension
        output = self.decoder_qnn.forward(qnn_input)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full autoencoding pipeline."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def train(self, data: np.ndarray, epochs: int = 50, learning_rate: float = 0.01) -> list[float]:
        """Very simple training loop that optimises the decoder QNN weights."""
        # For demonstration we use a classical optimizer (COBYLA) on the QNN parameters.
        # In this toy example the QNN has no trainable parameters, so the loop is a stub.
        history: list[float] = []
        for _ in range(epochs):
            loss = 0.0
            for x in data:
                recon = self.forward(x)
                loss += np.mean((x - recon) ** 2)
            loss /= len(data)
            history.append(loss)
            # No weight update in this placeholder
        return history


# --------------------------------------------------------------------------- #
#  PUBLIC API
# --------------------------------------------------------------------------- #
def QuantumAutoEncoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    trash_dim: int = 2,
    backend: str | None = None,
    shots: int = 1024,
    seed: int = 42,
) -> QuantumAutoEncoder:
    cfg = QuantumAutoEncoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        trash_dim=trash_dim,
        backend=backend,
        shots=shots,
        seed=seed,
    )
    return QuantumAutoEncoder(cfg)


__all__ = [
    "QuantumAutoEncoderFactory",
    "QuantumAutoEncoder",
    "QuantumAutoEncoderConfig",
]
