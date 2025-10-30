"""Quantum autoencoder using a swap‑test based variational circuit."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

class QuantumAutoencoder:
    """A variational autoencoder that uses a swap‑test to compare
    the encoded state with a reference state.  The circuit
    contains a RealAmplitudes ansatz that is later inverted,
    effectively implementing an encoder–decoder pair."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 2,
        backend: str | None = None,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.backend = backend or "qasm_simulator"
        self.qc = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(
            self.num_latent + 2 * self.num_trash + 1,
            name="q",
        )
        cr = ClassicalRegister(1, name="c")
        qc = QuantumCircuit(qr, cr)

        # variational encoder
        qc.append(
            RealAmplitudes(self.num_latent, reps=self.reps),
            range(self.num_latent),
        )

        # swap‑test with trash qubits
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)

        qc.measure(aux, cr[0])
        return qc

    def qnn(self) -> SamplerQNN:
        """Return a SamplerQNN that evaluates the probability of
        measuring a ``1`` on the ancilla qubit."""
        return SamplerQNN(
            circuit=self.qc,
            input_params=[],
            weight_params=self.qc.parameters,
            interpret=lambda x: x[0],
            output_shape=2,
            sampler=None,
        )

    def loss(self, params: np.ndarray) -> float:
        """Loss is one minus the probability of measuring 1."""
        qc = self.qc.copy()
        qc.set_parameters(params)
        qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=lambda x: x[0],
            output_shape=2,
            sampler=None,
        )
        probs = qnn(np.zeros((1,)))  # no input parameters
        return 1 - probs[0][0]

    def train(
        self,
        init_params: np.ndarray,
        *,
        maxiter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Optimize the circuit parameters with COBYLA."""
        opt = COBYLA()
        result = opt.minimize(
            self.loss,
            init_params,
            options={"maxiter": maxiter, "tol": tol},
        )
        return result.x

__all__ = ["QuantumAutoencoder"]
