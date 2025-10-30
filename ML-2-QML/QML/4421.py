import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.random import random_circuit
from qiskit import Aer

# ------------------------------------------------------------------
# Quantum sub‑circuits
# ------------------------------------------------------------------
def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Quantum auto‑encoder ansatz used in the hybrid sampler."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode block
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test style disentanglement
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


def _qfc_circuit(num_qubits: int) -> QuantumCircuit:
    """Quantum fully‑connected block that mimics the classical QFCModel."""
    qc = QuantumCircuit(num_qubits)
    # Random layer followed by a small trainable block
    qc.compose(RealAmplitudes(num_qubits, reps=3), inplace=True)
    for i in range(num_qubits):
        qc.rx(qiskit.circuit.Parameter(f"theta{i}"), i)
    return qc


def _sampler_circuit(inputs: int, weights: int) -> QuantumCircuit:
    """Parameterised sampler that produces a 2‑class probability."""
    inputs_vec = ParameterVector("in", inputs)
    weights_vec = ParameterVector("w", weights)
    qc = QuantumCircuit(inputs + weights)
    # Simple entangling pattern
    qc.ry(inputs_vec[0], 0)
    qc.ry(inputs_vec[1], 1)
    qc.cx(0, 1)
    for i, w in enumerate(weights_vec):
        qc.ry(w, i)
    qc.cx(0, 1)
    qc.ry(weights_vec[0], 0)
    qc.ry(weights_vec[1], 1)
    return qc


# ------------------------------------------------------------------
# Hybrid Sampler
# ------------------------------------------------------------------
class SamplerQNNGen099:
    """Quantum‑augmented sampler that concatenates an auto‑encoder,
    a QFC block, and a sampler sub‑circuit into a single variational
    circuit.  The circuit is wrapped by qiskit‑machine‑learning's
    SamplerQNN for differentiable sampling."""
    def __init__(self,
                 use_autoencoder: bool = True,
                 use_qfc: bool = True,
                 use_sampler: bool = True,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 num_qfc_qubits: int = 4,
                 sampler_weights: int = 4) -> None:
        self.use_autoencoder = use_autoencoder
        self.use_qfc = use_qfc
        self.use_sampler = use_sampler

        # Build the composite circuit
        qc = QuantumCircuit()

        if self.use_autoencoder:
            ae_circ = _autoencoder_circuit(num_latent, num_trash)
            qc.compose(ae_circ, inplace=True)

        if self.use_qfc:
            qfc_circ = _qfc_circuit(num_qfc_qubits)
            qc.compose(qfc_circ, inplace=True)

        if self.use_sampler:
            samp_circ = _sampler_circuit(inputs=2, weights=sampler_weights)
            qc.compose(samp_circ, inplace=True)

        # Parameters
        self.input_params = ParameterVector("in", 2)
        self.weight_params = ParameterVector("w", sampler_weights)

        # Sampler primitive
        sampler = StatevectorSampler()

        # Wrap into a SamplerQNN
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
            output_shape=2,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum sampler on the provided input vector."""
        return self.qnn.forward(inputs)


def SamplerQNNGen099_factory(**kwargs) -> SamplerQNNGen099:
    """Convenience factory mirroring the original API."""
    return SamplerQNNGen099(**kwargs)


__all__ = ["SamplerQNNGen099", "SamplerQNNGen099_factory"]
