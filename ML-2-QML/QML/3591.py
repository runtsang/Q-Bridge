"""Quantum self‑attention autoencoder using Qiskit.

The circuit first encodes classical data into a feature vector,
applies a self‑attention style rotation/entanglement block, then
passes the resulting state through a RealAmplitudes autoencoder
ansatz.  The interface mirrors the classical counterpart:
`SelfAttentionAutoencoder`.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning import algorithm_globals

# ----------------------------------------------------------------------
#  Helper to embed data
# ----------------------------------------------------------------------
def _feature_vector_circuit(data: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """Return a circuit that encodes `data` into the first `num_qubits` qubits."""
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)
    for i, val in enumerate(data):
        qc.ry(2 * np.arcsin(val), qr[i])  # simple amplitude encoding
    return qc

# ----------------------------------------------------------------------
#  Self‑attention block
# ----------------------------------------------------------------------
def _self_attention_block(num_qubits: int,
                          rotation_params: np.ndarray,
                          entangle_params: np.ndarray) -> QuantumCircuit:
    """Build a Qiskit self‑attention block.

    Parameters
    ----------
    num_qubits : int
        Number of qubits the block acts on.
    rotation_params : np.ndarray
        Shape (3*num_qubits,) – Rx, Ry, Rz angles per qubit.
    entangle_params : np.ndarray
        Shape (num_qubits-1,) – CRX angles between consecutive qubits.
    """
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    # Rotations
    for i in range(num_qubits):
        qc.rx(rotation_params[3 * i], qr[i])
        qc.ry(rotation_params[3 * i + 1], qr[i])
        qc.rz(rotation_params[3 * i + 2], qr[i])

    # Entanglement
    for i in range(num_qubits - 1):
        qc.crx(entangle_params[i], qr[i], qr[i + 1])

    return qc

# ----------------------------------------------------------------------
#  Quantum autoencoder ansatz
# ----------------------------------------------------------------------
def _autoencoder_ansatz(num_latent: int, num_trash: int) -> QuantumCircuit:
    """RealAmplitudes autoencoder with a swap‑test style readout."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    # Encode latent part
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=5),
               range(0, num_latent + num_trash), inplace=True)

    # Swap‑test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# ----------------------------------------------------------------------
#  Main hybrid circuit
# ----------------------------------------------------------------------
class SelfAttentionAutoencoder:
    """Quantum self‑attention followed by an autoencoder ansatz."""

    def __init__(self,
                 num_qubits: int,
                 num_latent: int,
                 num_trash: int,
                 *,
                 backend: qiskit.providers.backend.Backend | None = None) -> None:
        self.num_qubits = num_qubits
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = Sampler()
        self._build_qnn()

    def _build_qnn(self) -> None:
        """Construct the SamplerQNN used for inference."""
        # Placeholder parameters – will be overwritten by `run`
        self.qnn = SamplerQNN(
            circuit=QuantumCircuit(self.num_qubits),
            input_params=[],
            weight_params=[],
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> dict[str, int]:
        """
        Execute the full quantum circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for the attention block.
        entangle_params : np.ndarray
            Entanglement angles for the attention block.
        inputs : np.ndarray
            Classical input vector (length <= num_qubits).
        shots : int
            Number of shots for the sampler.

        Returns
        -------
        dict
            Measurement counts for the auxiliary qubit (0/1).
        """
        # Build data‑encoding circuit
        enc_circ = _feature_vector_circuit(inputs, self.num_qubits)

        # Self‑attention
        attn_circ = _self_attention_block(self.num_qubits,
                                          rotation_params,
                                          entangle_params)

        # Autoencoder ansatz
        ae_circ = _autoencoder_ansatz(self.num_latent, self.num_trash)

        # Combine all blocks
        full_circ = QuantumCircuit(self.num_qubits + 1)  # +1 for aux
        full_circ.compose(enc_circ, inplace=True)
        full_circ.compose(attn_circ, inplace=True)
        full_circ.compose(ae_circ, inplace=True)

        # Wrap in SamplerQNN
        self.qnn.circuit = full_circ
        self.qnn.weight_params = list(full_circ.parameters)

        # Execute
        result = self.sampler.run(full_circ, shots=shots).result()
        return result.get_counts(full_circ)

__all__ = ["SelfAttentionAutoencoder"]
