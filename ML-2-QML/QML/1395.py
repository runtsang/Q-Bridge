"""Quantum autoencoder with a variational encoder and classical decoder.

The :class:`Autoencoder__gen357` class implements a hybrid quantum‑classical autoencoder.
A RealAmplitudes ansatz encodes the input into a latent subspace via a swap‑test
mechanism.  The latent vector is extracted by sampling the expectation value of
a Pauli‑Z operator on the auxiliary qubit.  A simple linear decoder reconstructs
the input from the latent vector.  The quantum part is trained using a gradient‑based
optimizer on a statevector simulator.

Example
-------
>>> ae = Autoencoder__gen357(input_dim=5, latent_dim=3, num_trash=2)
>>> x = np.random.rand(4, 5)
>>> recon = ae.forward(x)
>>> recon.shape
(4, 5)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.utils import algorithm_globals
import torch
from torch import nn

class Autoencoder__gen357:
    """Quantum encoder + classical decoder autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 3,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.device = device

        algorithm_globals.random_seed = 42
        self.sampler = Sampler()

        # Build the variational circuit
        self.circuit = self._build_circuit()

        # Classical decoder: a simple linear layer implemented with torch
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the variational autoencoder circuit."""
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature map: encode input into first latent_dim qubits
        from qiskit.circuit import Parameter
        self.feature_params = [Parameter(f"θ_{i}") for i in range(self.latent_dim)]
        for i, param in enumerate(self.feature_params):
            qc.ry(param, qr[i])

        # Variational ansatz on latent+trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(self.latent_dim, self.latent_dim + self.num_trash), inplace=True)

        # Swap test with auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode classical data into latent vector via sampling."""
        latents = []
        for x in inputs:
            # Prepare circuit with input parameters
            qc = self.circuit.copy()
            param_bindings = {p: val for p, val in zip(self.feature_params, x[:self.latent_dim])}
            qc = qc.bind_parameters(param_bindings)
            # Sample expectation value of Z on auxiliary qubit
            result = self.sampler.run(qc).result()
            counts = result.get_counts()
            expectation = sum((1 if bit == "0" else -1) * c for bit, c in counts.items()) / sum(counts.values())
            latents.append(expectation)
        return np.array(latents).reshape(-1, 1)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Classical decoder mapping latent to reconstructed input."""
        return self.decoder(torch.from_numpy(latents)).detach().numpy()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        latents = self.encode(inputs)
        return self.decode(latents)

    def train(self, data: np.ndarray, epochs: int = 100, lr: float = 0.01) -> list[float]:
        """Train the hybrid autoencoder."""
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []

        for _ in range(epochs):
            optimizer.zero_grad()
            recon = torch.from_numpy(self.forward(data))
            loss = loss_fn(recon, torch.from_numpy(data))
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

__all__ = ["Autoencoder__gen357"]
