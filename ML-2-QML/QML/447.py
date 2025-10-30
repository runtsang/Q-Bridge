"""Quantum autoencoder using Qiskit with RealAmplitudes ansatz, domain‑wall injection, and hybrid training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


def _as_array(data: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    latent_dim: int = 3
    num_trash: int = 2
    reps: int = 5
    seed: int = 42


class AutoencoderNet:
    """Quantum autoencoder circuit with domain‑wall injection and variational decoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        algorithm_globals.random_seed = config.seed
        self.sampler = Sampler()
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Builds encode, decode, and domain‑wall circuits."""
        num_latent = self.config.latent_dim
        num_trash = self.config.num_trash
        num_qubits = num_latent + 2 * num_trash + 1

        # Encode ansatz
        self.encode_ansatz = RealAmplitudes(num_latent + num_trash, reps=self.config.reps)

        # Decoder ansatz (same as encode for symmetry)
        self.decode_ansatz = RealAmplitudes(num_latent + num_trash, reps=self.config.reps)

        # Domain‑wall circuit
        self.domain_wall = QuantumCircuit(num_qubits)
        for i in range(num_trash, num_qubit := num_latent + 2 * num_trash):
            self.domain_wall.x(i)

        # Full autoencoder circuit
        self.circuit = QuantumCircuit(num_qubits, 1)
        self.circuit.compose(self.domain_wall, inplace=True)
        self.circuit.compose(self.encode_ansatz, range(num_latent + num_trash), inplace=True)
        aux = num_latent + 2 * num_trash
        self.circuit.h(aux)
        for i in range(num_trash):
            self.circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, 0)

        # SamplerQNN for inference
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.encode_ansatz.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def encode(self, inputs: np.ndarray) -> Statevector:
        """Encode classical inputs into a quantum state via the sampler."""
        # Prepare input state vector
        input_sv = Statevector.from_label("0" * (self.config.latent_dim + self.config.num_trash))
        # In a full implementation, we would map inputs to rotation angles; here we mock
        return input_sv

    def decode(self, latents: np.ndarray) -> Statevector:
        """Decode latent representation back to classical space."""
        # For demonstration, we simply return a placeholder state
        return Statevector.from_label("0" * (self.config.latent_dim + self.config.num_trash))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Full forward pass: encode → decode → measurement."""
        # Encode input to latent
        latent_sv = self.encode(inputs)
        # Decode latent to reconstruction
        recon_sv = self.decode(latent_sv.data)
        # Sample measurement probabilities
        probs = self.sampler.run(self.circuit).result().get_counts()
        return np.array([probs.get("0", 0) / sum(probs.values()), probs.get("1", 0) / sum(probs.values())])

    def fidelity(self, target: Statevector, output: Statevector) -> float:
        """Fidelity between target and output states."""
        return abs(target.data @ output.data.conj()) ** 2


def Autoencoder() -> AutoencoderNet:
    """Convenience factory returning a quantum autoencoder."""
    config = AutoencoderConfig()
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
    early_stop_patience: int = 10,
) -> List[float]:
    """Hybrid training loop optimizing fidelity via COBYLA."""
    optimizer = COBYLA(maxfun=1000)
    target_sv = Statevector.from_label("0" * (model.config.latent_dim + model.config.num_trash))
    history: List[float] = []
    best_fid = -1.0
    patience = 0

    def objective(params: np.ndarray) -> float:
        # Update ansatz parameters
        for i, p in enumerate(params):
            model.encode_ansatz.params[i] = p
        # Forward pass
        probs = model.forward(data)
        # Convert probabilities to a statevector (simplified)
        output_sv = Statevector([probs[0], probs[1]])
        return -model.fidelity(target_sv, output_sv)

    for epoch in range(epochs):
        params = np.array([p for p in model.encode_ansatz.parameters])
        res = optimizer.minimize(objective, params)
        loss = -res.fun  # negative fidelity
        history.append(loss)
        if loss > best_fid:
            best_fid = loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
