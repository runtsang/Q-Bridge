"""
Quantum self‑attention built with a parameterised variational circuit.
The implementation extends the seed by adding multiple entangling layers and
providing both measurement counts and Z‑expectations, which can be used for
hybrid training pipelines.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


class SelfAttention:
    """
    Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of entangling layers. Each layer applies a sequence of
        controlled‑R_x gates between neighboring qubits.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """
        Construct the variational circuit with rotation and entanglement layers.
        The rotation parameters are grouped per qubit (3 angles per qubit).
        Entanglement parameters are laid out layer‑wise for adjacent qubits.
        """
        qc = QuantumCircuit(self.qr, self.cr)

        # Rotation layer
        for i in range(self.n_qubits):
            idx = 3 * i
            qc.rx(rotation_params[idx], i)
            qc.ry(rotation_params[idx + 1], i)
            qc.rz(rotation_params[idx + 2], i)

        # Entanglement layers
        for l in range(self.n_layers):
            for i in range(self.n_qubits - 1):
                idx = l * (self.n_qubits - 1) + i
                qc.crx(entangle_params[idx], i, i + 1)

        qc.barrier()
        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the circuit on the provided backend and return measurement counts.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

    def expectation_z(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ):
        """
        Return the expectation value of the Pauli‑Z operator for each qubit.
        Uses the state‑vector simulator for exact results.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        backend_sv = Aer.get_backend("statevector_simulator")
        job = execute(qc, backend_sv)
        sv = job.result().get_statevector(qc)
        expect = []
        for i in range(self.n_qubits):
            # Pauli‑Z expectation: (|0> amplitude)^2 - (|1> amplitude)^2
            proj = 0
            for idx, amp in enumerate(sv):
                if ((idx >> i) & 1) == 0:
                    proj += abs(amp) ** 2
                else:
                    proj -= abs(amp) ** 2
            expect.append(proj)
        return np.array(expect)


def SelfAttention():
    """
    Factory that preserves the original seed's callable interface.
    Returns a SelfAttention instance with 4 qubits and two entangling layers.
    """
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return SelfAttention(n_qubits=4, n_layers=2)


__all__ = ["SelfAttention"]
