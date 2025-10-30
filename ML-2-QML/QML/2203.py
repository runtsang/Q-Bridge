"""Quantum self‑attention + variational autoencoder.

The quantum module implements a self‑attention style circuit that acts as a
feature map for a variational autoencoder.  The same interface as the
classical version is preserved, enabling a side‑by‑side comparison.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector

# --------------------------------------------------------------------------- #
#  Self‑attention circuit
# --------------------------------------------------------------------------- #
def _self_attention_circuit(num_qubits: int, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
    """Build a small self‑attention style circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used for the feature map.
    rotation_params : np.ndarray
        Parameters for RX/RZ rotations per qubit.
    entangle_params : np.ndarray
        Parameters for controlled‑X gates between adjacent qubits.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    # Feature‑map rotations
    for i in range(num_qubits):
        qc.rx(rotation_params[3 * i], i)
        qc.rz(rotation_params[3 * i + 1], i)
        qc.rx(rotation_params[3 * i + 2], i)

    # Entangling layer
    for i in range(num_qubits - 1):
        qc.crx(entangle_params[i], i, i + 1)

    return qc


# --------------------------------------------------------------------------- #
#  Variational auto‑encoder circuit
# --------------------------------------------------------------------------- #
def _autoencoder_ansatz(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct a RealAmplitudes ansatz that will act as the encoder/decoder.

    The circuit is split into a latent block and a trash block; a swap‑test
    is used to read out the latent state.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode latent + trash qubits
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc


# --------------------------------------------------------------------------- #
#  QNN wrapper
# --------------------------------------------------------------------------- #
def QuantumSelfAttentionAutoencoder(num_qubits: int = 4, num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a SamplerQNN that implements quantum self‑attention followed by a variational auto‑encoder.

    Parameters
    ----------
    num_qubits : int
        Total qubits for the attention feature‑map.
    num_latent : int
        Size of the latent block in the auto‑encoder ansatz.
    num_trash : int
        Number of ancillary qubits used in the swap‑test.
    """
    # Random seed for reproducibility
    qiskit.set_backend("qasm_simulator")
    sampler = StatevectorSampler()

    # Build attention circuit
    rotation_params = np.random.uniform(0, 2 * np.pi, 3 * num_qubits)
    entangle_params = np.random.uniform(0, 2 * np.pi, num_qubits - 1)
    attention_circ = _self_attention_circuit(num_qubits, rotation_params, entangle_params)

    # Build auto‑encoder ansatz
    ae_circ = _autoencoder_ansatz(num_latent, num_trash)

    # Combine circuits
    qc = attention_circ.compose(ae_circ, inplace=True)

    # Wrap in a SamplerQNN
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=ae_circ.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["QuantumSelfAttentionAutoencoder"]
