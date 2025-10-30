"""Quantum autoencoder helper module.

This module supplies a parameterised Qiskit circuit that can be used as the
`quantum_encoder` callable in :class:`HybridAutoencoderNet`.  The design
combines the amplitude‑encoding + variational ansatz from the original
Autoencoder QML seed with the random‑layer / Pauli‑Z measurement strategy
from the Quantum‑NAT example.

Usage
-----
>>> from quantum_autoencoder import create_variational_autoencoder_circuit, simulate_quantum_latent
>>> circuit = create_variational_autoencoder_circuit(num_qubits=4, latent_dim=4, reps=3)
>>> latent = np.random.rand(1,4).astype(np.float32)
>>> out = simulate_quantum_latent(latent, circuit)
"""

import numpy as np
from typing import Callable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.algorithms import Sampler
from qiskit.quantum_info import Statevector
from qiskit.providers.fake_provider import FakeVigo

# -----------------------------------
# Helper: amplitude‑encoding circuit
# -----------------------------------
def _amplitude_encoding(data: np.ndarray, qr: QuantumRegister) -> None:
    """Load the data into the computational basis via amplitude encoding."""
    # Normalise to unit length
    raw = data.flatten()
    norm = np.linalg.norm(raw)
    if norm == 0:
        raise ValueError("Zero vector cannot be amplitude‑encoded.")
    state = raw / norm
    # Prepare statevector
    sv = Statevector(state, dims=2**len(qr))
    sv_data = sv.data
    # Convert to sequence of operations; here we use a simple state‑preparation
    # circuit via the built‑in method (fast, but not minimal).
    qc = QuantumCircuit(qr)
    qc.initialize(state, qr)
    # Append to target circuit
    for gate in qc:
        qr_circuit = gate.to_instruction()
        qr_circuit._name = gate.name
        qr_circuit._params = gate.params
        qr_circuit._qubits = gate.qubits
        qc.append(qr_circuit, gate.qubits, gate.clbits)
    return qc


# -----------------------------------
# Main variational autoencoder circuit
# -----------------------------------
def create_variational_autoencoder_circuit(
    num_qubits: int,
    latent_dim: int,
    reps: int = 5,
) -> QuantumCircuit:
    """
    Builds a variational circuit that:
    1. Amplitude‑encodes the latent vector into a quantum state.
    2. Applies a parameterised ansatz (RealAmplitudes).
    3. Measures all qubits in the computational basis.

    Parameters
    ----------
    num_qubits: int
        Number of qubits used for the autoencoder (must satisfy 2**num_qubits >= latent_dim).
    latent_dim: int
        Dimensionality of the classical latent vector.
    reps: int
        Number of repetitions in the RealAmplitudes ansatz.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.
    """
    if 2 ** num_qubits < latent_dim:
        raise ValueError("Number of qubits insufficient to encode the latent vector.")

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(qr)

    # 1. Amplitude encoding of the first `latent_dim` entries
    #    (the rest of the qubits are initialized to |0>)
    latent_placeholder = np.zeros(2 ** num_qubits, dtype=complex)
    latent_placeholder[:latent_dim] = 1.0 / np.sqrt(latent_dim)
    init_circ = QuantumCircuit(qr)
    init_circ.initialize(latent_placeholder, qr)
    qc.append(init_circ.to_instruction(), qr)

    # 2. Variational ansatz
    ansatz = RealAmplitudes(
        num_qubits=num_qubits, reps=reps, entanglement="full", insert_barriers=True
    )
    qc.append(ansatz.to_instruction(), qr)

    # 3. Measurement
    cr = ClassicalRegister(num_qubits, "c")
    qc.add_register(cr)
    qc.measure(qr, cr)

    return qc


# -----------------------------------
# Simulation helper
# -----------------------------------
def simulate_quantum_latent(
    latent: np.ndarray,
    circuit: QuantumCircuit,
    backend: Callable[[QuantumCircuit], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Runs the provided circuit with the supplied latent vector and returns
    the measurement expectation value as a real‑valued vector.

    Parameters
    ----------
    latent: np.ndarray
        Shape (batch_size, latent_dim) classical latent tensor.
    circuit: QuantumCircuit
        The variational autoencoder circuit built by
        :func:`create_variational_autoencoder_circuit`.
    backend: Callable[[QuantumCircuit], np.ndarray] | None
        Optional backend that accepts a circuit and returns a sample array.
        If ``None`` a simple classical simulator (FakeVigo) is used.

    Returns
    -------
    np.ndarray
        Shape (batch_size, num_qubits) array of expectation values in [-1, 1].
    """
    if backend is None:
        backend = FakeVigo().backend()

    batch_size, _ = latent.shape
    out = np.zeros((batch_size, circuit.num_qubits), dtype=np.float32)

    for i in range(batch_size):
        # Replace the placeholder in the circuit with the actual amplitude‑encoding
        # This is a lightweight hack: we re‑initialise the first qubits with the
        # current latent vector.
        circ = circuit.copy()
        # Build amplitude‑encoding for the current sample
        init_state = latent[i] / np.linalg.norm(latent[i])
        init_circ = QuantumCircuit(circ.qubits)
        init_circ.initialize(init_state, circ.qubits)
        circ.insert(0, init_circ.to_instruction())

        # Execute
        job = backend.run(circ, shots=1024)
        result = job.result()
        counts = result.get_counts()
        # Convert counts to expectation values
        exp_vals = np.zeros(circ.num_qubits, dtype=np.float32)
        for bitstring, n in counts.items():
            bits = np.array([int(b) for b in bitstring[::-1]])
            exp_vals += (1 - 2 * bits) * n  # +1 for 0, -1 for 1
        exp_vals /= 1024
        out[i] = exp_vals
    return out


# -----------------------------------
# Domain‑wall utility (from Autoencoder QML seed)
# -----------------------------------
def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Apply a domain wall (X gates) to qubits in the range [a, b)."""
    for i in range(a, b):
        circuit.x(i)
    return circuit


__all__ = [
    "create_variational_autoencoder_circuit",
    "simulate_quantum_latent",
    "domain_wall",
]
