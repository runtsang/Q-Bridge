"""
Quantum hybrid autoencoder that mirrors the classical structure using a variational circuit
and a QLayer‑style quantum fully‑connected block.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import RawFeatureVector

# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer (QLayer) – inspired by QuantumNAT
# --------------------------------------------------------------------------- #
def qlayer_circuit(num_wires: int = 4, reps: int = 2) -> QuantumCircuit:
    """
    Builds a small quantum circuit that performs a random layer followed by
    single‑qubit rotations and a controlled‑rotation, mirroring the QLayer
    from the QuantumNAT example.
    """
    qr = QuantumRegister(num_wires, "q")
    qc = QuantumCircuit(qr)
    # Random layer
    qc.h(qr)
    for i in range(num_wires):
        qc.rx(np.random.rand(), qr[i])
        qc.ry(np.random.rand(), qr[i])
    # Controlled‑rotation
    qc.crx(np.random.rand(), qr[0], qr[2])
    # Hadamard and Sx
    qc.h(qr[3])
    qc.sx(qr[2])
    qc.cx(qr[3], qr[0])
    return qc

# --------------------------------------------------------------------------- #
# Quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def auto_encoder_circuit(num_latent: int, num_trash: int, reps: int = 3) -> QuantumCircuit:
    """
    Variational autoencoder circuit that compresses a state into a latent subspace
    and reconstructs it using a swap‑test style measurement.
    """
    num_qubits = num_latent + 2 * num_trash + 1  # +1 auxiliary qubit
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz for latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
    qc.append(ansatz, list(range(num_latent + num_trash)))

    qc.barrier()

    # Swap‑test with auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# --------------------------------------------------------------------------- #
# Domain wall gadget (optional preprocessing)
# --------------------------------------------------------------------------- #
def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """
    Applies X gates to qubits in the range [a, b) to emulate a domain wall.
    """
    for i in range(a, b):
        circuit.x(i)
    return circuit

# --------------------------------------------------------------------------- #
# Hybrid quantum autoencoder construction
# --------------------------------------------------------------------------- #
def HybridAutoencoder() -> SamplerQNN:
    """
    Builds a SamplerQNN that combines:
      * a variational autoencoder circuit
      * a QLayer‑style quantum fully‑connected block
    The circuit is assembled by first applying a domain wall, then the
    autoencoder, followed by the qlayer. The output is interpreted as a
    2‑dimensional reconstruction vector.
    """
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    # Parameters for the autoencoder
    num_latent = 3
    num_trash = 2
    ae_circuit = auto_encoder_circuit(num_latent, num_trash, reps=3)

    # Domain wall preprocessing
    dw_circuit = domain_wall(QuantumCircuit(5), 0, 5)

    # QLayer circuit
    ql_circuit = qlayer_circuit(num_wires=4, reps=2)

    # Assemble full circuit
    full_qc = QuantumCircuit(num_latent + 2 * num_trash + 5 + 1, 1)
    full_qc.compose(dw_circuit, range(num_latent + num_trash), inplace=True)
    full_qc.compose(ae_circuit, range(num_latent + num_trash + 5), inplace=True)
    full_qc.compose(ql_circuit, range(num_latent + num_trash + 5 + 1), inplace=True)

    # Interpret the measurement outcome as a 2‑dimensional vector
    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=full_qc,
        input_params=[],
        weight_params=ae_circuit.parameters + ql_circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["HybridAutoencoder"]
