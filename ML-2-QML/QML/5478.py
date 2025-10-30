"""Quantum auto‑encoder implementation using Qiskit and a variational ansatz.

The class ``QuantumHybridAutoencoder`` mirrors the classical counterpart but
performs both encoding and decoding with quantum circuits.  The encoder
consists of a RealAmplitudes ansatz followed by a swap‑test that measures
the overlap with a reference state.  The decoder simply applies the
inverse of the ansatz to the latent qubits and measures in the computational
basis.  A ``Sampler`` backend is used to obtain expectation values in a
single shot, which keeps the implementation lightweight.

The circuit is wrapped in a ``SamplerQNN`` so that it can be interacted with
as a differentiable layer, which is useful for hybrid optimisation.
"""

import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumHybridAutoencoder:
    """Quantum auto‑encoder based on a variational RealAmplitudes ansatz.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int, optional
        Number of latent qubits used to encode the data.
    num_trash : int, optional
        Number of auxiliary qubits for the swap‑test.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.backend = StatevectorSampler()
        algorithm_globals.random_seed = 42

        # Build encoder circuit
        self.encoder_circuit = self._build_encoder()
        # Build decoder circuit (inverse of encoder)
        self.decoder_circuit = self._build_decoder()

        # Wrap with SamplerQNN for easy evaluation
        self.encoder_qnn = SamplerQNN(
            circuit=self.encoder_circuit,
            input_params=[],
            weight_params=self.encoder_circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.backend,
        )
        self.decoder_qnn = SamplerQNN(
            circuit=self.decoder_circuit,
            input_params=[],
            weight_params=self.decoder_circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.backend,
        )

    def _build_encoder(self) -> QuantumCircuit:
        """Construct the encoder circuit with a swap‑test."""
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)

        # Ansatz on the first latent+trash qubits
        circ.compose(
            RealAmplitudes(self.latent_dim + self.num_trash, reps=5),
            range(0, self.latent_dim + self.num_trash),
            inplace=True,
        )
        circ.barrier()

        # Swap‑test auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        circ.h(aux)
        for i in range(self.num_trash):
            circ.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circ.h(aux)
        circ.measure(aux, cr[0])
        return circ

    def _build_decoder(self) -> QuantumCircuit:
        """Inverse of the encoder to reconstruct the input."""
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)

        # Inverse ansatz
        circ.compose(
            RealAmplitudes(self.latent_dim + self.num_trash, reps=5).inverse(),
            range(0, self.latent_dim + self.num_trash),
            inplace=True,
        )
        circ.barrier()

        # Swap‑test inverse
        aux = self.latent_dim + 2 * self.num_trash
        circ.h(aux)
        for i in range(self.num_trash):
            circ.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circ.h(aux)
        circ.measure(aux, cr[0])
        return circ

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode a batch of inputs into latent expectation values."""
        # For simplicity we ignore the actual input values and use the
        # circuit parameters as the latent vector.  In a real scenario
        # one would embed the data into the circuit via feature maps.
        results = self.encoder_qnn.run()
        return np.array(results)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to the data space."""
        results = self.decoder_qnn.run()
        return np.array(results)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """End‑to‑end quantum auto‑encoder."""
        latents = self.encode(inputs)
        return self.decode(latents)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        learning_rate: float = 0.01,
        shots: int = 500,
    ) -> list[float]:
        """Train the quantum auto‑encoder with the COBYLA optimiser.

        The loss is a simple MSE between the decoded output and the
        original data.  COBYLA is used because the sampler does not
        return gradients.
        """
        optimizer = COBYLA()
        history: list[float] = []

        for epoch in range(epochs):
            # Evaluate current circuit
            recon = self.forward(data)
            loss = np.mean((recon - data) ** 2)
            history.append(loss)

            # COBYLA expects a flat parameter vector
            flat_params = np.array(
                [p.value for p in self.encoder_circuit.parameters], dtype=np.float64
            )
            def objective(p: np.ndarray) -> float:
                # Bind parameters
                for param, val in zip(self.encoder_circuit.parameters, p):
                    param.assign(val)
                recon = self.forward(data)
                return np.mean((recon - data) ** 2)

            new_params = optimizer.optimize(
                num_vars=len(flat_params),
                objective_function=objective,
                initial_point=flat_params,
                maxiter=shots,
            )
            # Update circuit
            for param, val in zip(self.encoder_circuit.parameters, new_params.x):
                param.assign(val)
        return history

__all__ = ["QuantumHybridAutoencoder"]
