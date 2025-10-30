"""
Quantum hybrid model that mimics the classical HybridConvAE.

The quantum pipeline first applies a parametric quanvolution
followed by a quantum autoencoder based on a sampler QNN.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

# ----------------------------------------------------------------------
# 1. Quantum convolution (quanvolution)
# ----------------------------------------------------------------------
class QuanvCircuit:
    """Parameterised circuit that emulates a convolution filter."""

    def __init__(self, kernel_size: int, threshold: float, shots: int = 1024):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        self.circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]

        # Encode classical pixel values into rotation angles
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)

        # Add a shallow random circuit to mix correlations
        self.circuit += random_circuit(self.n_qubits, depth=2)

        # Measure all qubits
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the quanvolution on a 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            Shape (kernel_size, kernel_size) with pixel intensities.

        Returns
        -------
        np.ndarray
            Array of probabilities of measuring |1> on each qubit.
        """
        flat = data.reshape(1, self.n_qubits)
        param_binds = [{t: np.pi if val > self.threshold else 0.0 for t, val in zip(self.theta, arr)}
                       for arr in flat]

        job = qiskit.execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            bits = np.array([int(b) for b in bitstring[::-1]])
            probs += bits * cnt
        probs /= self.shots * self.n_qubits
        return probs


# ----------------------------------------------------------------------
# 2. Quantum autoencoder circuit
# ----------------------------------------------------------------------
def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Construct a simple quantum autoencoder using a RealAmplitudes ansatz
    and a swap‑test style readout.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)

    # Swap‑test for latent extraction
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc


class HybridConvAE:
    """
    Quantum counterpart to the classical HybridConvAE.

    It performs a quanvolution followed by a quantum autoencoder
    implemented as a SamplerQNN.  The API is intentionally
    compatible with the classical version: :func:`run` accepts a
    2‑D numpy array and returns a reconstructed scalar.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 1024,
        num_latent: int = 3,
        num_trash: int = 2,
    ) -> None:
        self.quanv = QuanvCircuit(kernel_size, threshold, shots)
        self.backend = Aer.get_backend("aer_simulator_statevector")

        # Build the autoencoder QNN
        ae_circ = _autoencoder_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(
            circuit=ae_circ,
            input_params=[],
            weight_params=ae_circ.parameters,
            interpret=lambda x: x,  # identity
            output_shape=2,
            sampler=self.backend,
        )

        # Optimizer for variational parameters (placeholder)
        self.optimizer = COBYLA()

    def run(self, patch: np.ndarray) -> float:
        """
        Forward pass of the hybrid quantum model.

        Parameters
        ----------
        patch : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Reconstructed scalar value after the quantum autoencoder.
        """
        # Step 1: quanvolution
        prob_vec = self.quanv.run(patch)

        # Step 2: feed probabilities into QNN
        # The QNN expects a flat vector of probabilities as parameters
        # For simplicity, we bind the probabilities to the circuit weights
        param_binds = [{p: v for p, v in zip(self.qnn.weight_params, prob_vec)}]

        job = self.qnn.run(param_binds=param_binds, training=False)
        # The sampler returns a list of outcome counts; we take the mean
        # of the single classical bit measurement.
        values = [int(k) for k, _ in job]
        return float(np.mean(values))

    def train(self, data: np.ndarray, epochs: int = 10, learning_rate: float = 0.01) -> None:
        """
        Very light‑weight training loop that optimises the autoencoder
        parameters to minimise the reconstruction error of the quanvolution
        output.

        Parameters
        ----------
        data : np.ndarray
            Collection of image patches of shape (N, kernel_size, kernel_size).
        epochs : int
            Number of optimisation passes.
        learning_rate : float
            Step size for the COBYLA optimiser (used via the `maxfun` option).
        """
        algorithm_globals.random_seed = 42
        for epoch in range(epochs):
            losses = []
            for patch in data:
                target = self.quanv.run(patch)  # ground truth from quanvolution
                pred = self.run(patch)
                loss = (pred - target).item() ** 2
                losses.append(loss)

                # Simple gradient‑free update using COBYLA
                def objective(params):
                    self.qnn.weight_params = params
                    return loss

                self.optimizer.optimize(
                    len(self.qnn.weight_params), objective,
                    maxfun=10, maxiter=1, learning_rate=learning_rate
                )
            print(f"Epoch {epoch+1}/{epochs} | MSE: {np.mean(losses):.4f}")

__all__ = ["HybridConvAE"]
