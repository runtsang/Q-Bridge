"""
Hybrid self‑attention implemented with Qiskit.  The circuit
combines a rotation‑entanglement block (classical style) with a
parameter‑ized fully‑connected sub‑circuit for quantum feature
extraction.  The output is the expectation value of a Pauli‑Z
measurement, which serves as the attention weighting.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers import Backend

class HybridSelfAttentionQuantum:
    """
    Quantum self‑attention block that mirrors the classical
    HybridSelfAttention interface.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for the attention circuit.
    backend : qiskit.providers.Backend, optional
        The quantum backend to execute on.  Defaults to Aer qasm_simulator.
    shots : int, optional
        Number of shots for sampling.  Defaults to 1024.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 backend: Backend | None = None,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameters for rotation and entanglement
        self.rotation_params = Parameter("θ")
        self.entangle_params = Parameter("ϕ")

        # Parameter for the fully‑connected sub‑circuit
        self.fc_theta = Parameter("γ")

    def _build_circuit(self,
                       rotation_values: np.ndarray,
                       entangle_values: np.ndarray,
                       fc_value: float) -> QuantumCircuit:
        """Construct the full circuit for a single forward pass."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Rotation block (R_X, R_Y, R_Z per qubit)
        for i in range(self.n_qubits):
            circuit.rx(rotation_values[3 * i], i)
            circuit.ry(rotation_values[3 * i + 1], i)
            circuit.rz(rotation_values[3 * i + 2], i)

        # Entanglement block (controlled‑X between neighbours)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_values[i], i, i + 1)

        # Fully‑connected sub‑circuit on a single auxiliary qubit
        # that mimics a parameterised neural layer.
        aux = QuantumRegister(1, "aux")
        circuit.add_register(aux)
        circuit.h(aux[0])
        circuit.ry(fc_value, aux[0])

        # Measure all qubits
        circuit.measure_all()

        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            fc_param: float,
            shots: int | None = None) -> np.ndarray:
        """
        Execute the circuit and return the expectation value of
        a weighted measurement that serves as the attention score.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of length 3 * n_qubits for RX, RY, RZ angles.
        entangle_params : np.ndarray
            Array of length n_qubits - 1 for controlled‑X angles.
        fc_param : float
            Single real parameter for the fully‑connected sub‑circuit.
        shots : int, optional
            Override the default number of shots.

        Returns
        -------
        np.ndarray
            Expectation value of the measurement, shape (1,).
        """
        shots = shots or self.shots
        circuit = self._build_circuit(rotation_params, entangle_params, fc_param)

        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to expectation value of Pauli‑Z on the first qubit
        exp_z = 0.0
        for bitstring, cnt in counts.items():
            # bitstring order: last qubit first; we use the first qubit measured
            qubit_state = int(bitstring[-1])
            exp_z += (1 - 2 * qubit_state) * cnt
        exp_z /= shots

        return np.array([exp_z])

__all__ = ["HybridSelfAttentionQuantum"]
