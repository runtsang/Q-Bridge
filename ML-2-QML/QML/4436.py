"""Hybrid quantum kernel that fuses self‑attention, an auto‑encoder and a Ry
feature‑map.

The class :class:`HybridKernelMethod` mirrors the classical
``HybridKernelMethod`` but performs all operations on a quantum device.
It can compute exact kernel values from state‑vector overlaps and noisy
estimates via a swap‑test circuit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes

# --------------------------------------------------------------------------- #
# Classical‑style self‑attention circuit (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

# --------------------------------------------------------------------------- #
# Quantum auto‑encoder circuit (Qiskit)
# --------------------------------------------------------------------------- #
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a small quantum auto‑encoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit

# --------------------------------------------------------------------------- #
# Hybrid kernel implementation
# --------------------------------------------------------------------------- #
class HybridKernelMethod:
    """Quantum kernel that fuses self‑attention, an auto‑encoder and a Ry
    feature‑map.  Exact kernel values are obtained from state‑vector
    overlaps; noisy estimates use a swap‑test circuit.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024, backend=None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

    # --------------------------------------------------------------------- #
    # State preparation helpers
    # --------------------------------------------------------------------- #
    def _prepare_circuit(self, data: np.ndarray) -> QuantumCircuit:
        """Build a circuit that prepares the feature‑map state for a single
        data point.  The circuit is a concatenation of a self‑attention
        block, a small auto‑encoder and a Ry‑rotation feature‑map.
        """
        qc = QuantumCircuit(self.n_qubits)
        # Self‑attention
        sa = QuantumSelfAttention(self.n_qubits)
        rotation_params = np.random.randn(3 * self.n_qubits)
        entangle_params = np.random.randn(self.n_qubits - 1)
        qc.compose(sa._build_circuit(rotation_params, entangle_params), inplace=True)
        # Auto‑encoder (fixed parameters)
        ae = autoencoder_circuit(num_latent=3, num_trash=2)
        qc.compose(ae, inplace=True)
        # Feature‑map (Ry rotations)
        for i, val in enumerate(data):
            qc.ry(val, i)
        return qc

    # --------------------------------------------------------------------- #
    # Exact kernel evaluation
    # --------------------------------------------------------------------- #
    def _statevector(self, data: np.ndarray) -> Statevector:
        """Return the statevector of the prepared circuit."""
        qc = self._prepare_circuit(data)
        return Statevector.from_instruction(qc)

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the exact Gram matrix using state‑vector overlaps."""
        X_sv = [self._statevector(x) for x in X]
        Y_sv = [self._statevector(y) for y in Y]
        kernel = np.zeros((len(X), len(Y)))
        for i, sx in enumerate(X_sv):
            for j, sy in enumerate(Y_sv):
                # Fidelity = |<sx|sy>|^2
                kernel[i, j] = abs(sx @ sy.conjugate()) ** 2
        return kernel

    # --------------------------------------------------------------------- #
    # Noisy kernel estimation via swap‑test
    # --------------------------------------------------------------------- #
    def _swap_test(self, x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
        """Return a swap‑test circuit that estimates the fidelity between two
        feature‑map states.
        """
        n = self.n_qubits
        qr1 = QuantumRegister(n, "q1")
        qr2 = QuantumRegister(n, "q2")
        anc = QuantumRegister(1, "a")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr1, qr2, anc, cr)

        # Prepare the two states
        for i, val in enumerate(x):
            qc.ry(val, qr1[i])
        for i, val in enumerate(y):
            qc.ry(val, qr2[i])

        # Ancilla Hadamard
        qc.h(anc[0])

        # Controlled SWAP
        for i in range(n):
            qc.cswap(anc[0], qr1[i], qr2[i])

        # Hadamard
        qc.h(anc[0])
        qc.measure(anc[0], cr[0])
        return qc

    def kernel_matrix_with_noise(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return a noisy estimate of the Gram matrix using a swap‑test
        implementation with a finite number of shots.
        """
        shots = shots or self.shots
        rng = np.random.default_rng(seed)
        kernel = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                qc = self._swap_test(x, y)
                job = execute(qc, self.backend, shots=shots, seed_simulator=seed)
                counts = job.result().get_counts(qc)
                p0 = counts.get("0", 0) / shots
                fidelity = 2 * p0 - 1  # fidelity = |<x|y>|^2
                kernel[i, j] = fidelity
        return kernel

__all__ = [
    "HybridKernelMethod",
]
