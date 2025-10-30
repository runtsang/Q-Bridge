"""Quantum self‑attention with variational circuit and parameter‑shift gradients.

The circuit uses Ry/Rz rotations for the “query” and “key” embeddings and
controlled‑Rx gates for entanglement.  After measurement the expectation
values of Pauli‑Z on each qubit are turned into softmax attention weights.
Gradient estimation is available via the parameter‑shift rule.
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliExpectation, PauliSumOp, StateFn
from qiskit.opflow import Gradient
from qiskit.opflow import CircuitStateFn
from qiskit.opflow import PauliOp
from qiskit.opflow import StateFn, ExpectationFactory
from qiskit.opflow import ParameterResolver
from qiskit.quantum_info import Pauli
from qiskit.tools.visualization import plot_histogram
import numpy as np
from typing import Tuple


class SelfAttentionQuantum:
    """
    Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must equal embed_dim).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = AerSimulator()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Instantiate a parameterized circuit with rotation and entanglement."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circ = QuantumCircuit(qr, cr)

        # Rotation parameters
        self.rotation_params = ParameterVector("rot", 3 * self.n_qubits)
        # Entanglement parameters
        self.entangle_params = ParameterVector("ent", self.n_qubits - 1)

        for i in range(self.n_qubits):
            circ.rx(self.rotation_params[3 * i], i)
            circ.ry(self.rotation_params[3 * i + 1], i)
            circ.rz(self.rotation_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            circ.crx(self.entangle_params[i], i, i + 1)

        circ.measure(qr, cr)
        return circ

    def _parameter_resolver(self, rot_vals: np.ndarray, ent_vals: np.ndarray) -> ParameterResolver:
        """Map raw numpy arrays to ParameterResolver for the circuit."""
        resolver = ParameterResolver(
            {
                **{self.rotation_params[3 * i]: rot_vals[3 * i]
                   for i in range(self.n_qubits)},
                **{self.rotation_params[3 * i + 1]: rot_vals[3 * i + 1]
                   for i in range(self.n_qubits)},
                **{self.rotation_params[3 * i + 2]: rot_vals[3 * i + 2]
                   for i in range(self.n_qubits)},
                **{self.entangle_params[i]: ent_vals[i]
                   for i in range(self.n_qubits - 1)},
            }
        )
        return resolver

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a softmax over expectation values of Pauli‑Z.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles (shape 3*n_qubits).
        entangle_params : np.ndarray
            Entanglement angles (shape n_qubits-1).
        shots : int, default 1024
            Number of shots for measurement.

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_qubits,).
        """
        circ = self.circuit
        resolver = self._parameter_resolver(rotation_params, entangle_params)
        bound_circ = circ.bind_parameters(resolver)

        compiled = transpile(bound_circ, self.backend)
        qobj = assemble(compiled, shots=shots)
        result = self.backend.run(qobj).result()

        # Compute expectation values of Z on each qubit
        z_expect = []
        for i in range(self.n_qubits):
            # Pauli string Z on qubit i
            pauli = Pauli('I' * i + 'Z' + 'I' * (self.n_qubits - i - 1))
            op = PauliSumOp.from_pauli(pauli)
            expectation = PauliExpectation().convert(StateFn(bound_circ, state=True)).compose(op).eval().real
            z_expect.append(expectation)

        # Convert to softmax attention weights
        z_expect = np.array(z_expect)
        attn_weights = np.exp(z_expect) / np.sum(np.exp(z_expect))
        return attn_weights

    def parameter_shift_grad(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        epsilon: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate gradients of the expectation‑value based attention weights
        using the parameter‑shift rule.

        Parameters
        ----------
        rotation_params : np.ndarray
            Current rotation parameters.
        entangle_params : np.ndarray
            Current entanglement parameters.
        epsilon : float, default 1e-3
            Shift amount for the parameter‑shift rule.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients w.r.t. rotation and entanglement parameters.
        """
        rot_grad = np.zeros_like(rotation_params)
        ent_grad = np.zeros_like(entangle_params)

        # Rotation parameters
        for idx in range(len(rotation_params)):
            shift = np.zeros_like(rotation_params)
            shift[idx] = epsilon
            f_plus = self.run(rotation_params + shift, entangle_params)
            f_minus = self.run(rotation_params - shift, entangle_params)
            rot_grad[idx] = (f_plus - f_minus) / (2 * epsilon)

        # Entanglement parameters
        for idx in range(len(entangle_params)):
            shift = np.zeros_like(entangle_params)
            shift[idx] = epsilon
            f_plus = self.run(rotation_params, entangle_params + shift)
            f_minus = self.run(rotation_params, entangle_params - shift)
            ent_grad[idx] = (f_plus - f_minus) / (2 * epsilon)

        return rot_grad, ent_grad


__all__ = ["SelfAttentionQuantum"]
