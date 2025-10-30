"""Quantum autoencoder with a variational swap‑test and tunable entanglement."""

import json
import numpy as np
import warnings

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector


class AutoencoderGen246:
    """Quantum autoencoder combining a RealAmplitudes ansatz, a tunable entanglement block,
    and a variational swap‑test.  The QNN can be trained jointly with classical parameters.
    """

    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        entanglement_depth: int = 1,
        reps: int = 5,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.entanglement_depth = entanglement_depth
        self.reps = reps
        algorithm_globals.random_seed = 42
        self.sampler = Sampler()

    # --------------------------------------------------------------------------- #
    # Helper: domain wall insertion
    # --------------------------------------------------------------------------- #
    @staticmethod
    def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        """Apply X gates to qubits in the range [start, end)."""
        for i in range(start, end):
            circuit.x(i)
        return circuit

    # --------------------------------------------------------------------------- #
    # Helper: tunable entanglement block
    # --------------------------------------------------------------------------- #
    def _entanglement_block(self, qc: QuantumCircuit, qubits: list[int]) -> QuantumCircuit:
        """Apply a chain of CNOTs controlled by the entanglement depth."""
        depth = self.entanglement_depth
        for d in range(depth):
            for i in range(len(qubits) - 1):
                qc.cx(qubits[i], qubits[i + 1])
            # Reverse direction for next depth to increase connectivity
            qubits = qubits[::-1]
        return qc

    # --------------------------------------------------------------------------- #
    # Build the autoencoder circuit
    # --------------------------------------------------------------------------- #
    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode: RealAmplitudes ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Entanglement block on the remaining qubits
        entanglement_qubits = list(range(self.num_latent + self.num_trash, num_qubits - 1))
        qc = self._entanglement_block(qc, entanglement_qubits)

        # Swap‑test with auxiliary qubit
        aux = num_qubits - 1
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    # --------------------------------------------------------------------------- #
    # Create the SamplerQNN
    # --------------------------------------------------------------------------- #
    def get_qnn(self) -> SamplerQNN:
        qc = self._build_circuit()
        # No feature parameters: the circuit is fully parameterized by the ansatz
        qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=lambda x: x,  # identity interpretation
            output_shape=2,  # measurement outcome 0/1
            sampler=self.sampler,
        )
        return qnn

    # --------------------------------------------------------------------------- #
    # Training routine (COBYLA minimizer)
    # --------------------------------------------------------------------------- #
    def train(
        self,
        training_data: np.ndarray,
        epochs: int = 20,
        learning_rate: float = 1e-2,
    ) -> list[float]:
        """Train the QNN to reconstruct the training data using a simple MSE loss."""
        qnn = self.get_qnn()
        optimizer = COBYLA()
        loss_history: list[float] = []

        def loss_fn(params: np.ndarray) -> float:
            qnn.set_weights(params)
            # Sample from the circuit
            result = qnn.predict(training_data)
            # Convert probabilities to expected value
            preds = np.mean(result, axis=0)
            # Compute MSE against target (here we use training_data as target)
            loss = np.mean((preds - training_data) ** 2)
            return loss

        # Initialize parameters
        params = np.random.uniform(-np.pi, np.pi, size=len(qnn.parameters))
        for _ in range(epochs):
            params = optimizer.optimize(loss_fn, params, learning_rate=learning_rate)
            loss_history.append(loss_fn(params))
        return loss_history

__all__ = ["AutoencoderGen246"]
