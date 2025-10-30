import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN


def Autoencoder(latent_dim: int = 3, trash_dim: int = 2, reps: int = 2) -> SamplerQNN:
    """
    Quantum autoencoder that maps an input state to a latent subspace,
    performs a swap‑test with a reference, and decodes back.
    The circuit is built as a variational RealAmplitudes encoder
    followed by a domain‑wall encoding on the trash qubits.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        """
        Optional domain‑wall layer that flips a contiguous block of qubits.
        """
        for i in range(start, end):
            circuit.x(i)
        return circuit

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        total_qubits = num_latent + 2 * num_trash + 1  # +1 ancilla for swap‑test
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encoder: variational RealAmplitudes on the latent qubits
        circuit.compose(
            RealAmplitudes(num_latent, reps=reps), range(0, num_latent), inplace=True
        )
        circuit.barrier()

        # Domain‑wall on the first trash block
        circuit = domain_wall(circuit, num_latent, num_latent + num_trash)

        # Swap‑test with a reference state |0...0⟩ on the second trash block
        ancilla = num_latent + 2 * num_trash
        circuit.h(ancilla)
        for i in range(num_trash):
            circuit.cswap(ancilla, num_latent + i, num_latent + num_trash + i)
        circuit.h(ancilla)

        circuit.measure(ancilla, cr[0])
        return circuit

    circuit = auto_encoder_circuit(latent_dim, trash_dim)
    circuit.draw(output="mpl", style="clifford")

    # Define the interpret function to return the probability of measuring 0
    def interpret(x: np.ndarray) -> np.ndarray:
        return x[:, 0]  # probability of ancilla being 0

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


def train_qml_autoencoder(
    data: np.ndarray,
    latent_dim: int = 3,
    trash_dim: int = 2,
    epochs: int = 20,
    learning_rate: float = 0.01,
):
    """
    Simple training loop for the quantum autoencoder using COBYLA.
    The loss is the probability of measuring ancilla 0 (i.e., the fidelity with the
    reference state). The optimizer updates the variational parameters.
    """
    qnn = Autoencoder(latent_dim, trash_dim)
    opt = COBYLA(learning_rate=learning_rate)

    history = []

    for epoch in range(epochs):
        # The circuit has no explicit input; we feed a dummy vector of zeros
        # to let the optimizer update the parameters.
        dummy = np.zeros((1, 0))
        probs = qnn.predict(dummy)
        loss = 1.0 - probs[0]  # minimize 1 - P(0)
        opt.step(qnn.parameters, lambda p: loss)

        history.append(loss)
        print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.4f}")

    return history


__all__ = ["Autoencoder", "train_qml_autoencoder"]
