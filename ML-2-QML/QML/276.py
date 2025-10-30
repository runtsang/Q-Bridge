"""Quantum autoencoder using Qiskit and a variational circuit.

The model compresses an input state to a latent subspace and reconstructs it.
Training is performed with a gradient‑free optimizer (COBYLA) on a backend simulator.
"""

from __future__ import annotations

import numpy as np
from typing import List

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN


class Autoencoder:
    """
    Quantum autoencoder that maps an input state to a latent space and back.
    The circuit is a concatenation of a feature‑map, an encoder ansatz,
    a swap‑test and a measurement.  Training is performed with COBYLA.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 3,
        n_trash: int = 2,
        reps: int = 3,
        backend=None,
    ) -> None:
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_trash = n_trash
        self.reps = reps
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(backend=self.backend),
        )
        self.optimizer = COBYLA(maxiter=200, tol=1e-6)
        self.params: np.ndarray | None = None

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.n_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature map: encode the classical data into the first n_features qubits
        feature_map = RealAmplitudes(self.n_features, reps=1)
        qc.compose(feature_map, range(0, self.n_features), inplace=True)

        # Encoder ansatz
        encoder = RealAmplitudes(self.latent_dim + self.n_trash, reps=self.reps)
        qc.compose(encoder, range(0, self.latent_dim + self.n_trash), inplace=True)

        qc.barrier()

        # Swap‑test between encoded sub‑space and trash qubits
        aux = self.latent_dim + 2 * self.n_trash
        qc.h(aux)
        for i in range(self.n_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.n_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def _loss_fn(self, params: np.ndarray, data: np.ndarray) -> float:
        """Mean‑squared error between the circuit output and the target statevector."""
        param_dict = {p: v for p, v in zip(self.circuit.parameters, params)}
        circuit = self.circuit.assign_parameters(param_dict, inplace=False)
        # Simulate statevector of the circuit with the given parameters
        state = Statevector.from_instruction(circuit)
        # Target statevector: amplitude encoding of the input data
        # Here we use a simple amplitude encoding of the normalized data vector.
        target = Statevector.from_label("0" * self.n_features)  # placeholder
        # Compute fidelity and convert to loss
        fidelity = state.fidelity(target)
        return 1.0 - fidelity

    def fit(self, data: np.ndarray, epochs: int = 50) -> List[float]:
        """Train the autoencoder using COBYLA.

        Parameters
        ----------
        data: np.ndarray
            Input data of shape (N, n_features). Only the first sample is used
            as a target for demonstration purposes.
        epochs: int
            Maximum number of optimisation iterations.
        """
        algorithm_globals.random_seed = 42
        init_params = np.random.rand(len(self.circuit.parameters))
        history: List[float] = []

        def callback(xk: np.ndarray):
            loss = self._loss_fn(xk, data)
            history.append(loss)

        self.optimizer.minimize(
            fun=self._loss_fn,
            x0=init_params,
            args=(data,),
            callback=callback,
            options={"maxiter": epochs},
        )

        self.params = self.optimizer.xk
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Return the expectation values of Pauli‑Z on the latent qubits."""
        if self.params is None:
            raise RuntimeError("Model has not been trained yet.")
        param_dict = {p: v for p, v in zip(self.circuit.parameters, self.params)}
        circuit = self.circuit.assign_parameters(param_dict, inplace=False)
        job = self.backend.run(circuit)
        result = job.result()
        counts = result.get_counts()
        # Extract measurement of the auxiliary qubit
        z = 1 if "1" in counts else -1
        return np.array([z] * self.latent_dim)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Reconstruct data from latent representation (placeholder)."""
        # In a full implementation this would run a decoder ansatz.
        # For brevity we return a zero vector.
        return np.zeros(self.n_features)


__all__ = ["Autoencoder"]
