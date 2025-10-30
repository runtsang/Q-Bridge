"""Quantum autoencoder using Qiskit.

The encoder and decoder are variational circuits built from RealAmplitudes
ansatzes.  The input vector is encoded into a quantum state by applying
RY(2 * x) gates on the first set of qubits.  The encoder ansatz then
produces a compressed latent state on a subset of qubits.  The decoder
ansatz maps the latent qubits back to the full dimensionality and a
StatevectorSampler is used to obtain the reconstructed vector.  The
training loop optimizes the parameters of both ansatzes using a COBYLA
optimizer based on the mean‑squared error between the input and the
reconstructed vector.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA


class UnifiedAutoEncoder:
    """Quantum autoencoder with encoder/decoder ansatzes and a sampler.

    The circuits are defined once and parameters are updated during
    training.  The input vector is encoded into a quantum state by
    applying RY(2 * x) gates on each qubit; the variational ansatz then
    maps the state into a latent subspace.  The decoder ansatz reconstructs
    the full state.  The StatevectorSampler provides a classical preview
    of the quantum state for loss evaluation.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 ansatz_reps: int = 2,
                 shots: int = 1024) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = input_dim - latent_dim
        self.shots = shots

        # Circuits for the variational layers
        self.encoder_ansatz = RealAmplitudes(input_dim, reps=ansatz_reps)
        self.decoder_ansatz = RealAmplitudes(input_dim, reps=ansatz_reps)

        # Sampler for statevector extraction
        self.sampler = Sampler()

    def _build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Construct a full circuit for a single input vector."""
        qc = QuantumCircuit(self.input_dim)
        # Encode the classical data into the quantum state
        for idx, val in enumerate(x):
            angle = 2 * val  # simple linear mapping
            qc.ry(angle, idx)
        # Append variational ansatzes
        qc.compose(self.encoder_ansatz, inplace=True)
        qc.compose(self.decoder_ansatz, inplace=True)
        return qc

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct a single input vector."""
        qc = self._build_circuit(x)
        result = self.sampler.run([qc]).result()
        statevector = result.get_statevector(qc)
        # Interpret amplitude magnitudes as the reconstructed vector
        recon = np.abs(statevector) * np.linalg.norm(x)
        return recon

    def get_params(self) -> np.ndarray:
        """Flatten all ansatz parameters into a single vector."""
        return np.concatenate([self.encoder_ansatz.params, self.decoder_ansatz.params])

    def set_params(self, params: np.ndarray) -> None:
        """Set the ansatz parameters from a flattened vector."""
        split = len(self.encoder_ansatz.params)
        self.encoder_ansatz.params = params[:split]
        self.decoder_ansatz.params = params[split:]

    def loss(self, params: np.ndarray, data: np.ndarray) -> float:
        """Mean‑squared error over the dataset for a given set of params."""
        self.set_params(params)
        total = 0.0
        for x in data:
            recon = self.forward(x)
            total += np.mean((recon - x) ** 2)
        return total / data.shape[0]


def train_autoencoder_qml(model: UnifiedAutoEncoder,
                          data: np.ndarray,
                          *,
                          epochs: int = 50,
                          verbose: bool = False) -> list[float]:
    """Train the quantum autoencoder with COBYLA.

    Parameters
    ----------
    model
        Instance of :class:`UnifiedAutoEncoder`.
    data
        NumPy array of shape (N, input_dim) containing the training data.
    epochs
        Number of COBYLA optimization cycles.
    verbose
        If True, prints progress and loss values.
    """
    init_params = model.get_params()
    loss_history = [model.loss(init_params, data)]

    for epoch in range(epochs):
        optimizer = COBYLA(maxiter=200)
        opt_params, loss = optimizer.minimize(lambda p: model.loss(p, data), init_params)
        model.set_params(opt_params)
        loss_history.append(loss)
        if verbose:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.6f}")
        init_params = opt_params

    return loss_history


__all__ = [
    "UnifiedAutoEncoder",
    "train_autoencoder_qml",
]
