"""
Hybrid Quanvolution + Autoencoder – Quantum implementation.

Provides a variational autoencoder built on Qiskit's RealAmplitudes and a
swap‑test reconstruction, exposed as a SamplerQNN for easy integration with
PyTorch or scikit‑learn pipelines.
"""

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class QuantumAutoencoder:
    """
    Variational quantum autoencoder for small latent spaces.

    Parameters
    ----------
    num_qubits : int
        Total qubits in the circuit (latent + trash + auxiliary).
    latent_dim : int
        Number of qubits used to encode the latent representation.
    trash_dim : int, optional
        Number of qubits used as “trash” for the swap test. Default is 2.
    """

    def __init__(self, num_qubits: int, latent_dim: int, trash_dim: int = 2) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational ansatz over latent + trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        qc.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Swap test for reconstruction fidelity
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def get_qnn(self) -> SamplerQNN:
        """
        Returns a SamplerQNN that can be used as a drop‑in PyTorch module.
        """
        sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(1,),
            sampler=sampler,
        )

    def train(self, qnn: SamplerQNN, data: np.ndarray, *, epochs: int = 20) -> list[float]:
        """
        Very lightweight training loop using COBYLA on the sampler output.
        """
        opt = COBYLA()
        history: list[float] = []

        def loss_fn(params):
            qnn.update_weights(params)
            outputs = qnn.forward(data)
            # Simple MSE between output and target = 1 (perfect reconstruction)
            return np.mean((outputs - 1.0) ** 2)

        opt.minimize(loss_fn, qnn.get_weights(), epochs=epochs, tol=1e-4, disp=False)
        history.append(opt.last_fun_val)
        return history


__all__ = ["QuantumAutoencoder"]
