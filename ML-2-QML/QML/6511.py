"""AutoencoderGen345: quantum autoencoder using Qiskit and Qiskit Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

@dataclass
class AutoencoderGen345Config:
    """Configuration for the quantum autoencoder."""
    latent_qubits: int = 3
    trash_qubits: int = 2
    reps: int = 5
    backend: str = "statevector_simulator"

class AutoencoderGen345:
    """A hybrid quantum autoencoder that uses a variational encoder and a swap‑test decoder."""

    def __init__(self, config: AutoencoderGen345Config) -> None:
        self.config = config
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=(2,),
            sampler=Sampler(),
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs the variational encoder followed by a swap‑test decoder."""
        num_latent = self.config.latent_qubits
        num_trash = self.config.trash_qubits
        total = num_latent + 2 * num_trash + 1  # one auxiliary qubit for swap‑test
        qr = QuantumRegister(total, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational encoder
        encoder = RealAmplitudes(num_latent + num_trash, reps=self.config.reps)
        qc.append(encoder, range(0, num_latent + num_trash))

        # Swap‑test decoder
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _interpret(self, output: np.ndarray) -> np.ndarray:
        """Identity interpretation: return the raw measurement probability."""
        return output

    def get_qnn(self) -> SamplerQNN:
        """Return the underlying quantum neural network."""
        return self.qnn

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """Train the variational parameters using COBYLA to minimise mean‑squared error."""
        optimizer = COBYLA()
        history: List[float] = []

        def loss_fn(params: np.ndarray) -> float:
            # Set parameters
            self.qnn.set_weights(params)
            preds = []
            for _ in data:
                # For demonstration we ignore input encoding and sample from the circuit
                result = self.qnn.forward(np.array([0.0]))
                preds.append(result[0])
            preds = np.array(preds)
            # Target is the input itself (reconstruction)
            target = data
            return np.mean((preds - target) ** 2)

        # Initial parameters
        params = np.random.random(len(self.qnn.weights))
        for epoch in range(1, epochs + 1):
            params, loss, _ = optimizer.optimize(loss_fn, len(params), initial_point=params)
            history.append(loss)
            if callback:
                callback(epoch, loss)
        return history

__all__ = ["AutoencoderGen345", "AutoencoderGen345Config"]
