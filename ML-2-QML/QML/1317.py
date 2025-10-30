"""Quantum self‑attention using a variational circuit with tunable depth."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, PauliSumOp
from qiskit.quantum_info import Pauli


class SelfAttentionEnhanced:
    """
    Variational quantum self‑attention block.
    The circuit implements a feature‑mapping of the input followed by a depth‑controlled
    parameterized entangling layer.  Expectation values of a Pauli‑Z measurement
    are returned as the attention logits.
    """

    def __init__(self, n_qubits: int, depth: int = 2, backend: qiskit.providers.Backend = None):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits representing the input dimension.
        depth : int, optional
            Depth of the variational circuit. Defaults to 2.
        backend : qiskit.providers.Backend, optional
            Aer simulator or real backend. If None, AerSimulator('aer_simulator_statevector') is used.
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend or AerSimulator(method='statevector')
        # Parameterized feature map
        self.feature_map = RealAmplitudes(n_qubits, reps=1, entanglement='full')
        # Parameterized entangling circuit
        self.var_circuit = RealAmplitudes(n_qubits, reps=depth, entanglement='full')

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build the full circuit: feature map + variational circuit.
        rotation_params and entangle_params are flattened parameter arrays.
        """
        circuit = QuantumCircuit(self.n_qubits)
        # Feature map
        circuit.compose(self.feature_map.bind_parameters(rotation_params), inplace=True)
        # Variational circuit
        circuit.compose(self.var_circuit.bind_parameters(entangle_params), inplace=True)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the circuit and return expectation values of Z on each qubit.
        These expectation values can be interpreted as attention logits.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        # Measure expectation of Pauli-Z on each qubit
        expectation = PauliExpectation()
        # Build Pauli sum operator for Z on each qubit
        pauli_sum = PauliSumOp.from_list(
            [('I'*i + 'Z' + 'I'*(self.n_qubits-i-1), 1.0) for i in range(self.n_qubits)]
        )
        exp_val = expectation.convert(CircuitStateFn(pauli_sum, circuit))
        result = exp_val.eval().real
        return result


__all__ = ["SelfAttentionEnhanced"]
