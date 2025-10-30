"""FraudDetectionHybrid: Quantum variational circuit for fraud scoring.

The circuit is inspired by the photonic fraud‑detection seed but uses a
RealAmplitudes ansatz on a qubit backend.  The latent vector produced by a
classical autoencoder is fed into the circuit as rotation angles.  A
swap‑test with an auxiliary qubit yields a single expectation value that
serves as the fraud score.

The class exposes a lightweight `predict` method that runs the circuit on a
given backend (defaulting to the Aer qasm simulator) and returns the
measurement probability of the auxiliary qubit.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class FraudDetectionHybrid:
    """
    Quantum fraud‑detection circuit.

    Parameters
    ----------
    latent_dim : int
        Size of the latent vector produced by the classical autoencoder.
    num_trash : int, default 2
        Number of auxiliary qubits used for the swap‑test.
    shots : int, default 1024
        Number of shots for the measurement.
    backend : Backend, optional
        Quantum backend to execute the circuit.  If None, the Aer qasm_simulator
        is used.
    """

    def __init__(
        self,
        latent_dim: int,
        num_trash: int = 2,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        self.circuit, self.ansatz_params = self._build_circuit()
        self.sampler = Sampler(self.backend)

    def _build_circuit(self) -> tuple[QuantumCircuit, list]:
        """Construct the variational circuit with a swap‑test."""
        total_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz on latent + first trash
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=5)
        qc.compose(ansatz, list(range(self.latent_dim + self.num_trash)), inplace=True)

        # Swap‑test between latent and second trash
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)

        qc.measure(aux, cr[0])
        return qc, ansatz.params

    def predict(self, latent: Sequence[float]) -> float:
        """
        Evaluate the circuit for a single latent vector.

        Parameters
        ----------
        latent : Sequence[float]
            Latent vector of length `latent_dim`.

        Returns
        -------
        float
            Probability of measuring |1> on the auxiliary qubit.
        """
        if len(latent)!= self.latent_dim:
            raise ValueError("latent vector length mismatch")

        # Bind the latent angles to the ansatz parameters
        param_binds = {name: val for name, val in zip(self.ansatz_params, latent)}
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        prob_one = counts.get("1", 0) / self.shots
        return prob_one

    def __repr__(self) -> str:
        return f"<FraudDetectionHybrid latent_dim={self.latent_dim} shots={self.shots}>"

__all__ = ["FraudDetectionHybrid"]
