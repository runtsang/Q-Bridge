"""Quantum analogue of the ConvAutoencoder that replaces the 2×2 convolution
with a parameterized variational circuit and uses a swap‑test based
reconstruction.

The module is exposed via the same public factory `ConvAutoencoder()` to keep
API parity with the classical implementation.  Internally it builds a
`SamplerQNN` that takes 2×2 image patches, encodes each pixel as a rotation
on a qubit, applies a RealAmplitudes ansatz, and then performs a swap‑test
between a latent subspace and a trash subspace to obtain a reconstruction
probability.  The circuit can be executed on any Qiskit backend that
supports state‑vector sampling.

Typical usage:
```
qmodel = ConvAutoencoder(kernel_size=2, shots=512, backend="qasm_simulator")
output = qmodel.run(patch_array)
```
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector

# --------------------------------------------------------------------------- #
# Helper to build swap‑test circuit
# --------------------------------------------------------------------------- #
def _swap_test(num_qubits: int) -> QuantumCircuit:
    """Return a circuit that performs a swap‑test on `num_qubits`."""
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    # auxiliary qubit
    aux = num_qubits - 1
    qc.h(aux)
    for i in range(num_qubits - 1):
        qc.cswap(aux, i, i + 1)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class ConvAutoencoderQNN:
    """Variational quantum autoencoder that emulates a 2×2 convolution."""
    def __init__(
        self,
        kernel_size: int = 2,
        num_latent: int = 3,
        num_trash: int = 2,
        backend: str | None = None,
        shots: int = 256,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots

        # Feature vector encoding
        self.fv = RawFeatureVector(num_qubits=self.n_qubits)

        # Ansatz
        self.ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=5)

        # Swap‑test circuit
        self.swap_qc = _swap_test(self.n_qubits)

        # Full circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(self.fv, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)
        self.circuit.compose(self.swap_qc, inplace=True)

        # Sampler
        self.sampler = StatevectorSampler(backend=self.backend)

        # SamplerQNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.ansatz.parameters,
            interpret=lambda x: float(x[0]),   # single measurement bit
            output_shape=1,
            sampler=self.sampler,
        )

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) containing pixel values
            in the range [0, 255].

        Returns
        -------
        np.ndarray
            Reconstruction probability for the input patch.
        """
        # Normalize data to [0,1] and reshape
        vec = data.astype(np.float32).flatten() / 255.0
        # Bind parameters: each qubit receives an RX rotation proportional to pixel intensity
        param_binds = [{self.ansatz.params[i]: np.pi * vec[i] for i in range(self.n_qubits)}]
        result = self.qnn.run(param_binds, n_shots=self.shots)
        return np.asarray(result)

def ConvAutoencoder(
    kernel_size: int = 2,
    num_latent: int = 3,
    num_trash: int = 2,
    backend: str | None = None,
    shots: int = 256,
) -> ConvAutoencoderQNN:
    """Factory function mirroring the classical `ConvAutoencoder`."""
    return ConvAutoencoderQNN(
        kernel_size=kernel_size,
        num_latent=num_latent,
        num_trash=num_trash,
        backend=backend,
        shots=shots,
    )

__all__ = ["ConvAutoencoderQNN", "ConvAutoencoder"]
