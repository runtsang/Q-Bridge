"""Hybrid autoencoder implemented purely with Qiskit.

The quantum encoder mirrors the PennyLane version but is expressed with
Qiskit primitives.  It supports a layered structure inspired by the fraud‑detection
photonic program and a fully‑connected quantum layer.  A simple classical
decoder (numpy linear layer) is also provided for end‑to‑end reconstruction.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import PauliExpectation, StateFn, PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.utils import QuantumInstance


class HybridAutoencoder:
    """Hybrid autoencoder with a Qiskit variational encoder and a numpy decoder."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        reps: int = 3,
        seed: int | None = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.reps = reps
        self.seed = seed
        np.random.seed(seed)
        self.simulator = AerSimulator(method="statevector")
        self.qinstance = QuantumInstance(self.simulator, shots=None)

        # Build parameter vector for each layer: RX and RZ per qubit
        self.param_vectors: list[ParameterVector] = []
        for r in range(reps):
            self.param_vectors.append(ParameterVector(f"theta_{r}", length=num_qubits * 2))
        # Flattened parameter array for easy manipulation
        self.params = np.random.randn(reps * num_qubits * 2)

        # Classical decoder: linear weights (latent_dim -> input_dim)
        self.decoder_weights: np.ndarray | None = None

    def _build_circuit(self, input_angles: np.ndarray) -> QuantumCircuit:
        """Return a parameterised circuit that encodes the input and applies variational layers."""
        qubits = QuantumRegister(self.num_qubits, "q")
        circ = QuantumCircuit(qubits)

        # Input encoding via RX gates
        for i, angle in enumerate(input_angles):
            circ.rx(angle, qubits[i % self.num_qubits])

        # Variational layers
        param_idx = 0
        for r in range(self.reps):
            # RX and RZ gates
            for q in range(self.num_qubits):
                circ.rx(self.param_vectors[r][param_idx], qubits[q])
                param_idx += 1
                circ.rz(self.param_vectors[r][param_idx], qubits[q])
                param_idx += 1
            # Entanglement chain
            for q in range(self.num_qubits - 1):
                circ.cx(qubits[q], qubits[q + 1])
            circ.cx(qubits[-1], qubits[0])  # wrap‑around

        return circ

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode a batch of inputs into latent vectors.

        Parameters
        ----------
        inputs : np.ndarray, shape (batch, features)
            Input data.  Features are mapped to RX angles.
        Returns
        -------
        np.ndarray, shape (batch, latent_dim)
            Expectation values of PauliZ for each qubit.
        """
        batch = inputs.shape[0]
        latent = np.zeros((batch, self.latent_dim))
        for i in range(batch):
            circ = self._build_circuit(inputs[i])
            bound_circ = circ.bind_parameters(self._params_to_dict())
            result = self.simulator.run(bound_circ, shots=None).result()
            state = Statevector(result.get_statevector(circ))
            # Expectation of PauliZ on each qubit
            for q in range(self.latent_dim):
                op = PauliSumOp.from_list([("Z" * q) + ("I" * (self.num_qubits - q - 1), 1)])
                exp = PauliExpectation()(StateFn(op, is_measurement=True) @ StateFn(state)).eval().real
                latent[i, q] = exp
        return latent

    def _params_to_dict(self) -> dict:
        """Map the flat params array to the ParameterVector objects."""
        mapping: dict = {}
        idx = 0
        for r, vec in enumerate(self.param_vectors):
            n = len(vec)
            mapping.update({vec[i]: self.params[idx + i] for i in range(n)})
            idx += n
        return mapping

    def get_params(self) -> np.ndarray:
        """Return current variational parameters."""
        return self.params.copy()

    def set_params(self, new_params: np.ndarray) -> None:
        """Set variational parameters."""
        if new_params.shape!= self.params.shape:
            raise ValueError("Parameter shape mismatch.")
        self.params[:] = new_params

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Reconstruct input from latent vectors using a linear decoder."""
        if self.decoder_weights is None:
            raise RuntimeError("Decoder weights not initialized; call `train_decoder` first.")
        return latents @ self.decoder_weights.T

    def train_decoder(self, latents: np.ndarray, targets: np.ndarray, lr: float = 1e-3, epochs: int = 500):
        """Learn a linear decoder that maps latent vectors back to inputs."""
        # Simple gradient descent
        self.decoder_weights = np.random.randn(self.latent_dim, targets.shape[1]) * 0.1
        for _ in range(epochs):
            preds = self.decode(latents)
            grad = 2 * (preds - targets).T @ latents / targets.shape[0]
            self.decoder_weights -= lr * grad

    def train_encoder(
        self,
        data: np.ndarray,
        decoder_lr: float = 1e-3,
        encoder_lr: float = 1e-3,
        epochs: int = 200,
    ) -> list[float]:
        """Jointly train the encoder and decoder to minimise reconstruction error.

        Returns
        -------
        list[float]
            History of reconstruction loss.
        """
        # Initialize decoder
        batch = data[0]
        latent = self.encode(batch[None, :])[0]
        self.train_decoder(latent[None, :], batch, lr=decoder_lr, epochs=10)

        history = []
        for _ in range(epochs):
            total_loss = 0.0
            for x in data:
                # Encode
                latent = self.encode(x[None, :])[0]
                # Decode
                recon = self.decode(latent[None, :])[0]
                loss = np.mean((recon - x) ** 2)
                total_loss += loss
                # Gradient w.r.t decoder
                grad_decoder = 2 * (recon - x) * latent / latent.shape[0]
                self.decoder_weights -= decoder_lr * grad_decoder
                # Gradient w.r.t encoder via finite differences
                eps = 1e-5
                grad_enc = np.zeros_like(self.params)
                for i in range(len(self.params)):
                    orig = self.params[i]
                    self.params[i] = orig + eps
                    latent_eps = self.encode(x[None, :])[0]
                    recon_eps = self.decode(latent_eps[None, :])[0]
                    loss_eps = np.mean((recon_eps - x) ** 2)
                    grad_enc[i] = (loss_eps - loss) / eps
                    self.params[i] = orig
                self.params -= encoder_lr * grad_enc
            history.append(total_loss / len(data))
        return history


__all__ = ["HybridAutoencoder"]
