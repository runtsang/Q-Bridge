import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

def Autoencoder():
    """Hybrid quantum‑classical autoencoder that shares latent space with a dense classical encoder."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # ------------------------------------------------------------------
    # Classical dense encoder (used only for latent extraction)
    # ------------------------------------------------------------------
    def classical_encoder(x: np.ndarray) -> np.ndarray:
        """Fast numpy‑based encoder that maps input to latent dimension."""
        W = np.random.randn(x.shape[1], 32) * 0.1
        bias = np.zeros(32)
        return np.tanh(W @ x.T + bias).T

    # ------------------------------------------------------------------
    # Quantum circuit that maps latent to reconstruction
    # ------------------------------------------------------------------
    def auto_encoder_circuit(num_latent: int, num_trash: int):
        """Variational circuit with a swap‑test style entanglement to produce a quantum state."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(RealAmplitudes(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        # swap test
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)
        circuit.measure(cells=cr[0])
        return circuit

    # ------------------------------------------------------------------
    # Build the hybrid circuit: classical latent + quantum reconstruction
    # ------------------------------------------------------------------
    num_latent = 32
    num_trash = 2
    latent_circuit = auto_encoder_circuit(num_latent, num_trash)

    # ------------------------------------------------------------------
    # SamplerQNN wrapper
    # ------------------------------------------------------------------
    def identity_interpret(x: np.ndarray) -> np.ndarray:
        """Map quantum amplitudes to an output vector of size 1."""
        return x

    qnn = SamplerQNN(
        circuit=latent_circuit,
        input_params=[],
        weight_params=latent_circuit.parameters,
        interpret=identity_interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn
