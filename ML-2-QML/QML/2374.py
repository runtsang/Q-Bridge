# Autoencoder__gen178.py - Quantum part
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

# ----------------------------------------------------------------------
# Quantum encoder – a variational circuit producing a latent vector
# ----------------------------------------------------------------------
def quantum_encoder_circuit(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """
    Constructs a RealAmplitudes ansatz that takes a state vector of size 2**num_qubits
    and outputs a parameterised state.  The circuit is used as the encoder
    in the hybrid autoencoder.
    """
    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(qr)
    qc.compose(RealAmplitudes(num_qubits, reps=reps), inplace=True)
    return qc

# ----------------------------------------------------------------------
# Quantum decoder – swap‑test based reconstruction
# ----------------------------------------------------------------------
def quantum_decoder_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Implements a swap‑test that compares the encoded state with a reference
    state.  The ancilla qubit is measured to give a fidelity estimate.
    """
    qr = QuantumRegister(num_qubits + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Reference state |0…0> – implicit
    anc = qr[num_qubits]  # ancilla
    qc.h(anc)
    for i in range(num_qubits):
        qc.cswap(anc, i, i)  # trivial swap; placeholder for actual reconstruction logic
    qc.h(anc)
    qc.measure(anc, cr[0])
    return qc

# ----------------------------------------------------------------------
# Sampler‑based QNN wrapper
# ----------------------------------------------------------------------
def HybridQuantumAutoencoder(num_qubits: int, latent_dim: int) -> SamplerQNN:
    """
    Builds a SamplerQNN that maps a classical input vector to a quantum state
    of size 2**num_qubits, then applies the encoder circuit and returns
    the latent vector as a classical output.  The decoder is omitted here
    because the hybrid model will use the classical decoder.
    """
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    # The input parameters are the raw classical vector flattened into a state.
    # We use a dummy parameter list so that the QNN can be instantiated.
    input_params = [f"x{i}" for i in range(2 ** num_qubits)]
    weight_params = [f"w{i}" for i in range(2 ** num_qubits)]

    # Build the full circuit: encode the input state then apply the ansatz
    qc = quantum_encoder_circuit(num_qubits)
    qc.compose(RealAmplitudes(num_qubits), inplace=True)  # additional layer
    # The output is the statevector; we interpret it as a real vector
    def interpret(state: Statevector) -> np.ndarray:
        return np.real(state.data)

    qnn = SamplerQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        interpret=interpret,
        output_shape=(latent_dim,),
        sampler=sampler,
    )
    return qnn

__all__ = ["quantum_encoder_circuit", "quantum_decoder_circuit", "HybridQuantumAutoencoder"]
