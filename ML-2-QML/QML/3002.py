"""Quantum convolutional autoencoder.

The ConvAutoencoder class builds a parameterized quantum circuit that
first encodes a 2‑D input patch via RX rotations, then applies a
RealAmplitudes ansatz to learn a latent representation, and finally
performs a SWAP‑test style decoder that measures reconstruction
fidelity.  The `run` method accepts a 2‑D array and returns the
reconstruction error (1‑fidelity) computed by a state‑vector sampler.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class ConvAutoencoder:
    """Quantum convolutional autoencoder combining a parameterized
    convolution filter with a variational autoencoder ansatz.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_shots: int = 100,
        conv_threshold: float = 0.5,
        latent_dim: int = 3,
        trash_dim: int = 2,
        sampler_shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.conv_threshold = conv_threshold
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.backend = Aer.get_backend("qasm_simulator")
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[f"theta{i}" for i in range(self.n_qubits)],
            weight_params=self.circuit.parameters[self.n_qubits:],
            interpret=lambda x: x,
            output_shape=1,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.n_qubits + self.latent_dim + self.trash_dim
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Input encoding with RX rotations
        for i in range(self.n_qubits):
            qc.rx(Parameter(f"theta{i}"), i)

        qc.barrier()

        # Variational autoencoder ansatz on latent+trash qubits
        qc.compose(
            RealAmplitudes(self.latent_dim + self.trash_dim, reps=5),
            range(self.n_qubits, total_qubits),
            inplace=True,
        )

        qc.barrier()

        # SWAP‑test style decoder
        aux = total_qubits
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.n_qubits + i, self.n_qubits + self.latent_dim + i)
        qc.h(aux)

        qc.measure(aux, cr[0])

        return qc

    def run(self, data: np.ndarray | list[list[float]]) -> float:
        """Evaluate the quantum autoencoder on a single patch and return
        reconstruction error (1‑fidelity).
        """
        # Prepare input parameter bindings
        data = np.reshape(data, (self.n_qubits,))
        param_binds = {}
        for i, val in enumerate(data):
            param_binds[f"theta{i}"] = np.pi if val > self.conv_threshold else 0.0

        # Execute the SamplerQNN
        result = self.qnn(param_binds)

        # `result` is a 1‑D array of expectation values; we interpret
        # the first element as fidelity between input and reconstructed state.
        fidelity = float(result[0])
        error = 1.0 - fidelity
        return error

__all__ = ["ConvAutoencoder"]
