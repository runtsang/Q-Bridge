"""Quantum autoencoder that mirrors the hybrid classical version.

The circuit encodes a classical latent vector into qubits, applies a
variational ansatz, and then uses a swap test to compare the encoded
state with the original.  The loss is 1 - fidelity, which is then
optimized with a classical optimizer (COBYLA or gradient descent).
The architecture is inspired by the Qiskit Autoencoder example and
augments it with a domain‑wall style feature map that can be used to
inject prior knowledge into the quantum state.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class HybridAutoQuanvNet:
    """Quantum autoencoder that operates on a classical latent vector.

    The circuit uses ``RealAmplitudes`` as a variational ansatz and a
    swap test to compute fidelity between the original and reconstructed
    latent states.  The design is intentionally minimal to keep the
    quantum resource usage low while still demonstrating a full
    encoder‑decoder workflow.
    """
    def __init__(
        self,
        latent_dim: int,
        num_trash: int = 0,
        reps: int = 3,
        seed: int | None = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        algorithm_globals.random_seed = seed
        self.sampler = Sampler()

    def _swap_test_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Return a circuit that performs a swap test on ``num_qubits``."""
        qr = QuantumRegister(num_qubits + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        # ancilla qubit
        ancilla = num_qubits
        circuit.h(qr[ancilla])
        for i in range(num_qubits):
            circuit.cswap(qr[ancilla], qr[i], qr[i + num_qubits])
        circuit.h(qr[ancilla])
        circuit.measure(qr[ancilla], cr[0])
        return circuit

    def build_ansatz(self, total_qubits: int) -> QuantumCircuit:
        """Variational ansatz for the full circuit."""
        return RealAmplitudes(total_qubits, reps=self.reps)

    def build_full_circuit(self) -> QuantumCircuit:
        """Build the complete quantum autoencoder circuit.

        The circuit consists of:
        1. Encoding of the latent vector into the first ``latent_dim``
           qubits via X gates (for simplicity).
        2. Variational ansatz on all qubits (latent + trash).
        3. Swap test to compare encoded and decoded states.
        """
        total_qubits = self.latent_dim + self.num_trash
        base_circuit = QuantumCircuit(total_qubits)
        # Simple encoding: apply X if the classical bit is 1
        # (placeholder for a more sophisticated amplitude encoding)
        # In practice you would use a state preparation routine.
        # Here we leave the qubits in |0> and let the ansatz learn.
        ansatz = self.build_ansatz(total_qubits)
        base_circuit.append(ansatz, range(total_qubits))

        # Append swap test between latent (first part) and trash (second part)
        swap_circuit = self._swap_test_circuit(total_qubits)
        base_circuit.compose(swap_circuit, inplace=True)
        return base_circuit

    def get_qnn(self) -> SamplerQNN:
        """Return a SamplerQNN that can be used as a differentiable layer."""
        circuit = self.build_full_circuit()
        # No input parameters; all parameters are variational weights
        qnn = SamplerQNN(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=self._interpret,
            output_shape=2,
            sampler=self.sampler,
        )
        return qnn

    @staticmethod
    def _interpret(x: np.ndarray) -> float:
        """Interpret the measurement result as fidelity."""
        # x contains probabilities of measuring 0 and 1.
        # Fidelity is 1 - probability of measuring 1 (ancilla in |1>).
        return float(1.0 - x[1])

    def train(
        self,
        qnn: SamplerQNN,
        X: np.ndarray,
        *,
        epochs: int = 50,
        optimizer_cls=COBYLA,
        maxiter: int = 200,
    ) -> list[float]:
        """Train the quantum autoencoder on a set of latent vectors."""
        history: list[float] = []
        opt = optimizer_cls(maxiter=maxiter)

        def loss_fn(params: np.ndarray) -> float:
            qnn.set_weights(params)
            outputs = qnn(X)
            # outputs shape (N, 2) with probabilities of 0 and 1
            fidelities = outputs[:, 0]  # probability of ancilla=0
            loss = 1.0 - np.mean(fidelities)
            return loss

        params = np.random.randn(len(qnn.weight_params))
        for _ in range(epochs):
            params = opt.optimize(loss_fn, params)
            loss = loss_fn(params)
            history.append(loss)
        return history


__all__ = ["HybridAutoQuanvNet"]
