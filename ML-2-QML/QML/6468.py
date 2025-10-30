import numpy as np
from typing import Callable
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.utils import algorithm_globals

def _apply_domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply a domain‑wall pattern (X gates) to a range of qubits."""
    for qubit in range(start, end):
        circuit.x(qubit)
    return circuit

def quantum_autoencoder_circuit(num_latent: int,
                                num_trash: int,
                                reps: int = 5,
                                seed: int = 42) -> QuantumCircuit:
    """Construct a variational quantum encoder with a swap test for latent extraction."""
    algorithm_globals.random_seed = seed
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode data into the first num_latent qubits with a Hadamard to create superposition.
    circuit.h(range(num_latent))

    # Variational ansatz on the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Domain‑wall on the trash qubits to break symmetry
    _apply_domain_wall(circuit, num_latent, num_latent + num_trash)

    # Swap test to entangle latent with trash
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit

def get_quantum_encoder(input_dim: int,
                        latent_dim: int,
                        num_trash: int = 2,
                        reps: int = 5,
                        seed: int = 42) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that maps input vectors to quantum latent vectors."""
    sampler = Sampler()
    circuit = quantum_autoencoder_circuit(latent_dim, num_trash, reps=reps, seed=seed)

    def encoder(x: np.ndarray) -> np.ndarray:
        """Encode a batch of inputs into latent vectors."""
        batch_latent = []
        for vec in x:
            # Binarize input to 0/1 bits for basis state encoding.
            bits = (vec > 0.5).astype(int)
            circ = circuit.copy()
            for i, bit in enumerate(bits[:latent_dim]):
                if bit:
                    circ.x(i)
            result = sampler.run(circ).result()
            counts = result.get_counts()
            # Pick the most frequent measurement outcome.
            outcome = max(counts, key=counts.get)
            # Convert binary string to integer vector.
            latent_bits = np.array([int(b) for b in outcome[::-1]], dtype=int)
            batch_latent.append(latent_bits)
        return np.stack(batch_latent, axis=0)

    return encoder

__all__ = ["quantum_autoencoder_circuit", "get_quantum_encoder"]
