"""Quantum autoencoder with swap‑test and domain‑wall encoding."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper: domain‑wall gate construction
# --------------------------------------------------------------------------- #
def domain_wall_circuit(num_qubits: int, start: int, end: int) -> QuantumCircuit:
    """
    Inserts X gates on qubits in the range [start, end) to create a domain wall.
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(start, end):
        qc.x(i)
    return qc


# --------------------------------------------------------------------------- #
# Core autoencoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(
    num_latent: int,
    num_trash: int,
    domain_wall_range: Optional[Tuple[int, int]] = None,
) -> QuantumCircuit:
    """
    Builds a swap‑test based quantum autoencoder:
    - `num_latent`: qubits holding the latent representation.
    - `num_trash`: auxiliary qubits for the swap test.
    - `domain_wall_range`: optional, defines a domain wall on a sub‑segment.
    """
    total_q = num_latent + 2 * num_trash + 1  # +1 for ancilla
    qr = QuantumRegister(total_q, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode using a variational ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=4)
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)

    # Domain‑wall insertion if requested
    if domain_wall_range is not None:
        dw_qc = domain_wall_circuit(total_q, *domain_wall_range)
        qc.compose(dw_qc, inplace=True)

    # Swap test
    anc = num_latent + 2 * num_trash
    qc.h(anc)
    for i in range(num_trash):
        qc.cswap(anc, num_latent + i, num_latent + num_trash + i)
    qc.h(anc)
    qc.measure(anc, cr[0])

    return qc


# --------------------------------------------------------------------------- #
# SamplerQNN wrapper
# --------------------------------------------------------------------------- #
def create_quantum_autoencoder(
    num_latent: int,
    num_trash: int,
    domain_wall_range: Optional[Tuple[int, int]] = None,
    sampler: Optional[Sampler] = None,
) -> SamplerQNN:
    """
    Instantiates a SamplerQNN that implements the quantum autoencoder.
    The network outputs a single bit (the swap‑test result) which can be
    interpreted as a similarity score between the input and its reconstruction.
    """
    algorithm_globals.random_seed = 42
    qc = quantum_autoencoder_circuit(num_latent, num_trash, domain_wall_range)

    # No trainable input parameters; all weights are variational
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,  # raw measurement
        output_shape=1,
        sampler=sampler or Sampler(),
    )
    return qnn


# --------------------------------------------------------------------------- #
# Example optimisation routine
# --------------------------------------------------------------------------- #
def optimise_quantum_autoencoder(
    qnn: SamplerQNN,
    data: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> list[float]:
    """
    Optimises the variational parameters of the quantum autoencoder
    using COBYLA to maximise the swap‑test probability.
    """
    optimizer = COBYLA()
    history: list[float] = []

    def objective(theta: np.ndarray) -> float:
        qnn.set_weights(theta)
        probs = qnn.predict(data)
        # Swap‑test probability of measuring 1
        p1 = probs[:, 1].mean()
        loss = -p1  # maximise probability
        history.append(loss)
        return loss

    optimizer.minimize(
        objective,
        initial_point=np.random.uniform(0, 2 * np.pi, size=len(qnn.parameters)),
        maxiter=max_iter,
        tol=tol,
    )
    return history


__all__ = [
    "domain_wall_circuit",
    "quantum_autoencoder_circuit",
    "create_quantum_autoencoder",
    "optimise_quantum_autoencoder",
]
