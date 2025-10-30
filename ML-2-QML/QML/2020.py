import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator

__all__ = ["QuantumAutoencoder", "create_variational_circuit", "sample_latent_vector"]

def create_variational_circuit(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """Create a RealAmplitudes ansatz for the quantum encoder."""
    return RealAmplitudes(num_qubits, reps=reps)

def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply a domain wall (X gates) from start to end‑1."""
    for i in range(start, end):
        circuit.x(i)
    return circuit

def QuantumAutoencoder(
    num_latent: int,
    num_trash: int,
    reps: int = 3,
    backend: AerSimulator | None = None,
) -> QuantumCircuit:
    """Build a quantum autoencoder circuit with swap‑test style latent extraction."""
    backend = backend or AerSimulator()
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode part
    encoder = create_variational_circuit(num_latent + num_trash, reps=reps)
    qc.append(encoder, list(range(0, num_latent + num_trash)))

    # Swap‑test for latent extraction
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    # Optionally add a domain wall on the trash qubits
    trash_qubits = list(range(num_latent, num_latent + num_trash))
    domain_wall = QuantumCircuit(num_trash, name="domain_wall")
    domain_wall.x(range(num_trash))
    qc.append(domain_wall, trash_qubits)

    return qc

def sample_latent_vector(
    circuit: QuantumCircuit,
    input_state: np.ndarray,
    shots: int = 1024,
    backend: AerSimulator | None = None,
) -> np.ndarray:
    """Simulate the circuit and return the measured bitstring as a latent vector."""
    backend = backend or AerSimulator()
    # Prepare the state
    init_circuit = QuantumCircuit(circuit.num_qubits)
    init_circuit.initialize(input_state, list(range(circuit.num_qubits)))
    full_circuit = init_circuit.compose(circuit)
    result = backend.run(full_circuit, shots=shots).result()
    counts = result.get_counts()
    # Convert counts to a binary vector average
    vector = np.zeros(circuit.num_qubits, dtype=float)
    for bitstring, count in counts.items():
        bits = np.array([int(b) for b in bitstring[::-1]])
        vector += bits * count
    vector /= shots
    return vector
