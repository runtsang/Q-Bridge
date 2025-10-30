"""QuantumNATEnhanced: pure‑quantum implementation using Qiskit."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RX, RY, RZ, CRX
from qiskit.circuit import ParameterVector


class QuantumNATEnhanced:
    """Quantum‑only model that mirrors the hybrid architecture.
    The interface is intentionally similar to the PyTorch version:
    ``forward(inputs)`` returns a 4‑dimensional vector per batch element.
    Internally we perform angle‑encoding of a 4‑dimensional feature slice,
    apply a small variational circuit (self‑attention style) and measure all
    qubits to obtain a Z‑Pauli expectation value vector.
    """
    def __init__(self, input_dim: int = 4, shots: int = 1024):
        self.input_dim = input_dim
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.n_qubits = 4

        # Parameters for the variational block (self‑attention style)
        self.attn_params = ParameterVector("a", 12)  # 3 rotations per qubit
        self.ent_params  = ParameterVector("e", 3)   # 1 CRX per adjacent pair

        # Build the variational sub‑circuit
        self.var_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            self.var_circuit.append(RX(self.attn_params[3 * i]), [i])
            self.var_circuit.append(RY(self.attn_params[3 * i + 1]), [i])
            self.var_circuit.append(RZ(self.attn_params[3 * i + 2]), [i])
        for i in range(self.n_qubits - 1):
            self.var_circuit.append(CRX(self.ent_params[i]), [i, i + 1])
        self.var_circuit.measure_all()

    def angle_encode(self, features: np.ndarray) -> QuantumCircuit:
        """Encode a 4‑dimensional feature vector into qubit rotation angles."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i, coeff in enumerate(features):
            qc.rx(coeff, i)
            qc.ry(coeff, i)
            qc.rz(coeff, i)
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the full quantum circuit for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, input_dim).  The first four components are
            encoded into the qubits; the remaining are ignored.
        """
        results = []
        for feat in inputs:
            qc = self.angle_encode(feat[:self.n_qubits])
            qc += self.var_circuit
            job = execute(qc, self.backend, shots=self.shots)
            counts = job.result().get_counts(qc)
            # Convert measurement counts to expectation values of Pauli‑Z
            exp = np.mean([self._bitstring_to_z(b) * c for b, c in counts.items()])
            results.append(exp)
        return np.array(results)

    @staticmethod
    def _bitstring_to_z(bitstring: str) -> float:
        """Map a bitstring to a Pauli‑Z expectation value."""
        return 1.0 if bitstring.count('1') % 2 == 0 else -1.0


__all__ = ["QuantumNATEnhanced"]
