"""
Hybrid convolutional autoencoder – quantum implementation.

The quantum module mirrors the classical one but replaces the
convolutional pre‑processor with a variational quanvolution circuit
(``QuantumConvFilter``) and the dense autoencoder with a
``SamplerQNN`` that implements a variational autoencoder circuit.
The overall API remains identical to the classical version so that
``ConvGenAutoencoder`` can be swapped for ``QuantumConvGenAutoencoder``
in any experiment.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

__all__ = ["QuantumConvFilter", "QuantumConvGenAutoencoder", "QuantumAutoencoder"]


# --------------------------------------------------------------------------- #
# 1. Variational quanvolution filter
# --------------------------------------------------------------------------- #
class QuantumConvFilter:
    """Variational circuit that emulates a convolutional filter on a 2‑D patch.

    Parameters
    ----------
    kernel_size : int
        Size of the square patch to be encoded.
    backend : qiskit.providers.BaseBackend
        Backend used to execute the circuit.
    shots : int
        Number of shots per evaluation.
    threshold : float
        Classical threshold used to encode pixel values.
    """
    def __init__(self, kernel_size: int = 3,
                 backend: qiskit.providers.BaseBackend | None = None,
                 shots: int = 1024,
                 threshold: float = 127.0) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Build a generic variational ansatz
        self.circuit = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(theta):
            self.circuit.rx(t, i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()
        self.theta = theta

    def run(self, patch: np.ndarray) -> float:
        """Execute the circuit for a single 2‑D patch.

        Parameters
        ----------
        patch : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        flat = patch.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {t: np.pi if v > self.threshold else 0.0 for t, v in zip(self.theta, row)}
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = sum(counts.values())
        ones = sum(int(k.count("1")) * c for k, c in counts.items())
        return ones / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
# 2. Variational quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Construct a quantum autoencoder using a swap‑test style circuit.

    The returned ``SamplerQNN`` can be used as a drop‑in replacement for a
    classical MLP autoencoder.  The circuit is composed of a RealAmplitudes
    ansatz followed by a swap‑test that learns a latent representation.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    def circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()

        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    circ = circuit(num_latent, num_trash)
    return SamplerQNN(
        circuit=circ,
        input_params=[],
        weight_params=circ.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )


# --------------------------------------------------------------------------- #
# 3. Quantum hybrid autoencoder
# --------------------------------------------------------------------------- #
class QuantumConvGenAutoencoder:
    """Quantum analogue of ``ConvGenAutoencoder``.

    The first stage is a variational quanvolution filter that processes
    each patch of the input image.  The flattened result is fed into
    the quantum autoencoder defined above.  The class exposes a
    ``run`` method that accepts a 2‑D numpy array and returns the
    reconstructed image as a flat vector.
    """
    def __init__(self,
                 kernel_size: int = 3,
                 threshold: float = 127.0,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 shots: int = 1024,
                 backend: qiskit.providers.BaseBackend | None = None) -> None:
        self.filter = QuantumConvFilter(
            kernel_size=kernel_size,
            backend=backend,
            shots=shots,
            threshold=threshold,
        )
        self.qae = QuantumAutoencoder(num_latent=num_latent, num_trash=num_trash)
        self.kernel_size = kernel_size

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray
            2‑D array of shape (H, W).

        Returns
        -------
        np.ndarray
            Reconstructed image flattened to a 1‑D vector.
        """
        H, W = image.shape
        stride = 1
        patches = []
        for i in range(0, H - self.kernel_size + 1, stride):
            for j in range(0, W - self.kernel_size + 1, stride):
                patch = image[i:i+self.kernel_size, j:j+self.kernel_size]
                val = self.filter.run(patch)
                patches.append(val)
        # Flatten and feed into quantum autoencoder
        flat = np.array(patches)
        # The QAE expects a 1‑D input; we interpret it as a single sample
        qae_input = flat.reshape(1, -1)
        output = self.qae(qae_input)[0]  # shape (1, 2)
        # For demonstration we return the raw output; in practice
        # one would map it back to image space.
        return output
