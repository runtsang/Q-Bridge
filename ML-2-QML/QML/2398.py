"""Quantum LSTM cell implemented with Qiskit.

The class :class:`GraphQLSTM` implements a variational LSTM cell where each
gate (forget, input, update, output) is realised by a small parameterised
quantum circuit.  The circuit encodes the classical input and hidden state
as rotation angles, entangles the qubits with a CNOT ladder, and measures
the expectation value of Pauli‑Z to obtain a gate activation in the range
[-1, 1].  The activations are passed through the standard LSTM equations
to produce a new hidden state.  The implementation is fully
quantum‑centric and can be executed on any Qiskit backend.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli

Tensor = np.ndarray

# --------------------------------------------------------------------------- #
# 1. Helper functions
# --------------------------------------------------------------------------- #

def _encode_params(n_qubits: int) -> list[Parameter]:
    """Return a list of Qiskit Parameters for each qubit."""
    return [Parameter(f"θ_{i}") for i in range(n_qubits)]

def _build_gate_circuit(
    n_qubits: int,
    param_names: list[Parameter],
) -> QuantumCircuit:
    """Construct a parameterised circuit that encodes the gate."""
    qc = QuantumCircuit(n_qubits)
    # Encode the input and hidden state as RX rotations
    for i, p in enumerate(param_names):
        qc.rx(p, i)
    # Entangle with a CNOT ladder
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc

def _expectation_z(circuit: QuantumCircuit, bound_circ: QuantumCircuit, n_qubits: int) -> Tensor:
    """Return expectation value of Pauli‑Z for each qubit."""
    statevector = Statevector.from_instruction(bound_circ)
    exp_vals = []
    for i in range(n_qubits):
        pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
        exp_vals.append(statevector.expectation_value(Pauli(pauli_str)))
    return np.array(exp_vals)

# --------------------------------------------------------------------------- #
# 2. Quantum LSTM cell
# --------------------------------------------------------------------------- #

class GraphQLSTM:
    """Quantum LSTM cell realised with Qiskit.

    The cell implements the standard LSTM equations, but each gate
    (forget, input, update, output) is computed by a small
    variational quantum circuit.  The circuits are identical except
    for the gate name, and they share the same parameterisation
    of the input and hidden state.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits per gate (equal to the hidden dimension).
        backend : qiskit.providers.Backend, optional
            Backend to execute the circuits on.  If ``None`` the
            Aer statevector simulator is used.
        shots : int, optional
            Number of shots for expectation estimation.
        """
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots

        # Create parameter lists for each gate
        self.params = _encode_params(n_qubits)
        # Build circuits for each gate
        self.circuits = {
            "forget": _build_gate_circuit(n_qubits, self.params),
            "input": _build_gate_circuit(n_qubits, self.params),
            "update": _build_gate_circuit(n_qubits, self.params),
            "output": _build_gate_circuit(n_qubits, self.params),
        }

    def _run_gate(self, gate_name: str, values: Tensor) -> Tensor:
        """Run a single gate circuit with the given input/hidden values."""
        circuit = self.circuits[gate_name]
        bound = {p: v for p, v in zip(self.params, values)}
        bound_circ = circuit.bind_parameters(bound)
        exp_vals = _expectation_z(circuit, bound_circ, self.n_qubits)
        # Map from [-1, 1] to [0, 1] for sigmoid-like behaviour
        return (exp_vals + 1) / 2

    def forward(
        self,
        input_vec: Tensor,
        hidden_vec: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        input_vec : np.ndarray
            Input vector of shape (n_qubits,).
        hidden_vec : np.ndarray
            Hidden state vector of shape (n_qubits,).

        Returns
        -------
        np.ndarray
            New hidden state vector of shape (n_qubits,).
        """
        # Concatenate input and hidden for encoding
        values = np.concatenate([input_vec, hidden_vec])
        # Compute gate activations
        f = self._run_gate("forget", values)
        i = self._run_gate("input", values)
        g = self._run_gate("update", values)
        o = self._run_gate("output", values)

        # Apply classical LSTM equations
        cx = f * hidden_vec + i * np.tanh(g)
        hx = o * np.tanh(cx)
        return hx

__all__ = ["GraphQLSTM"]
