"""Hybrid quantum autoencoder that couples a variational circuit with a classical fully‑connected post‑processing layer."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# 1. Classical post‑processing layer
# --------------------------------------------------------------------------- #
class ClassicalFCL:
    """
    Tiny feed‑forward layer that maps the 2‑dimensional quantum output
    to a single scalar.  It mirrors the behaviour of the classical
    fully‑connected layer from the reference pair.
    """
    def __init__(self, n_features: int = 2) -> None:
        self.weights = np.random.randn(n_features, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.weights)

# --------------------------------------------------------------------------- #
# 2. Variational autoencoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build the core variational circuit used by the SamplerQNN."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Ansatz: RealAmplitudes on the first (num_latent + num_trash) qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test with an auxiliary qubit
    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary)
    circuit.measure(auxiliary, cr[0])

    return circuit

# --------------------------------------------------------------------------- #
# 3. Domain‑wall augmentation
# --------------------------------------------------------------------------- #
def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Apply a chain of X‑gates to qubits a..b‑1."""
    for i in range(a, b):
        circuit.x(i)
    return circuit

# --------------------------------------------------------------------------- #
# 4. Hybrid quantum autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderHybrid:
    """
    Variational autoencoder that uses a SamplerQNN and a classical
    fully‑connected layer for interpretation.  The circuit implements
    a swap‑test based autoencoder with a domain‑wall injection.
    """
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Core circuit
        self.circuit = quantum_autoencoder_circuit(num_latent, num_trash)

        # Inject domain‑wall pattern
        self.circuit = domain_wall(self.circuit, 0, num_latent + num_trash)

        # SamplerQNN wrapper
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],  # no classical inputs
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # raw probability vector
            output_shape=2,
            sampler=Sampler(backend=self.backend, shots=self.shots),
        )

        # Classical post‑processing
        self.fcl = ClassicalFCL(n_features=2)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the hybrid autoencoder on a batch of parameter vectors.
        Parameters are bound to the quantum circuit; the resulting
        measurement probabilities are fed through the classical FCL.
        """
        # Bind parameters
        param_binds = [{p: t for p, t in zip(self.circuit.parameters, thetas)}]
        probs = self.qnn.forward(param_binds)  # shape (1, 2)
        return self.fcl(probs)

def make_autoencoder_hybrid(
    num_latent: int = 3,
    num_trash: int = 2,
    backend=None,
    shots: int = 1024,
) -> AutoencoderHybrid:
    """Convenience constructor mirroring the classical factory."""
    return AutoencoderHybrid(num_latent, num_trash, backend, shots)

__all__ = [
    "AutoencoderHybrid",
    "make_autoencoder_hybrid",
    "ClassicalFCL",
]
