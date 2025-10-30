"""Quantum autoencoder implementation using Qiskit and qiskit‑machine‑learning."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Callable, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
# 1.  Quantum autoencoder class
# --------------------------------------------------------------------------- #
class QuantumAutoencoder:
    """
    Variational quantum autoencoder that maps a high‑dimensional input state
    to a low‑dimensional latent subspace.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        trash_dim: int = 2,
        reps: int = 3,
        backend: str = "aer_simulator",
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the classical input vector (must be a power of two).
        latent_dim : int
            Number of qubits used for the latent representation.
        trash_dim : int
            Number of ancillary qubits for swap‑test reconstruction.
        reps : int
            Number of repetitions for the RealAmplitudes ansatz.
        backend : str
            Backend name for simulation.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.backend = backend

        self._circuit = self._build_circuit()
        self._sampler = Sampler(self.backend)
        self.qnn = SamplerQNN(
            circuit=self._circuit,
            input_params=[],
            weight_params=self._circuit.parameters,
            interpret=self._identity_interpret,
            output_shape=(self.latent_dim,),
            sampler=self._sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Build the variational autoencoder circuit."""
        total_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode classical input as amplitude‑encoded state
        raw_feat = RawFeatureVector(self.input_dim)
        qc.compose(raw_feat, range(0, self.input_dim), inplace=True)

        # Variational ansatz on latent + first trash block
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=self.reps)
        qc.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Swap‑test between latent and second trash block
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)

        # Measure auxiliary qubit to obtain a fidelity proxy
        qc.measure(aux, cr[0])
        return qc

    @staticmethod
    def _identity_interpret(x: np.ndarray) -> np.ndarray:
        """Identity interpretation for the sampler output."""
        return x

    def loss_function(self, params: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the loss as 1 - fidelity between current and target latent states.
        """
        self.qnn.set_weights(params)
        output = self.qnn.forward([])
        sv = Statevector(output)
        fidelity = sv.fidelity(Statevector(target))
        return 1.0 - fidelity

    def train(
        self,
        target_latent: np.ndarray,
        *,
        epochs: int = 50,
        learning_rate: float = 0.01,
        optimizer_cls: Callable = COBYLA,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Train the variational parameters to match the target latent state.

        Parameters
        ----------
        target_latent : np.ndarray
            One‑dimensional array of length 2^latent_dim representing the desired
            latent state in amplitude‑encoding.
        epochs : int
            Number of optimization iterations.
        learning_rate : float
            Step size for gradient‑free optimizers.
        optimizer_cls : Callable
            Optimizer class from qiskit_machine_learning.optimizers.
        verbose : bool
            Print progress.

        Returns
        -------
        params : np.ndarray
            Optimized circuit parameters.
        history : list[float]
            Loss history.
        """
        opt = optimizer_cls(method="cobyla", maxiter=epochs, tol=1e-6)
        history: List[float] = []

        def objective(p):
            loss = self.loss_function(p, target_latent)
            history.append(loss)
            if verbose:
                print(f"Epoch {len(history):03d} | Loss: {loss:.6f}")
            return loss

        opt.minimize(objective, x0=np.random.rand(len(self.qnn.weight_params)))
        return opt.x, history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode classical data to a latent quantum state.

        Parameters
        ----------
        data : np.ndarray
            Classical input vector of shape (input_dim,).

        Returns
        -------
        counts : np.ndarray
            Sampling counts of the auxiliary qubit which proxy the fidelity.
        """
        qc = self._build_circuit()
        raw = RawFeatureVector(self.input_dim)
        qc.compose(raw, range(0, self.input_dim), inplace=True)
        result = self._sampler.run(qc)
        return result.get_counts()

def Autoencoder() -> QuantumAutoencoder:
    """
    Convenience factory that returns a ready‑to‑train quantum autoencoder
    with toy dimensions.
    """
    # Example dimensions for a toy problem: 4‑dim input → 2‑dim latent
    input_dim = 4  # 2 qubits
    latent_dim = 2  # 1 qubit latent
    return QuantumAutoencoder(input_dim, latent_dim)
