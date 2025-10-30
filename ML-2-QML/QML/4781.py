"""Hybrid quantum self‑attention that combines a quanvolution filter with a
variational attention circuit.  The implementation follows the FastBaseEstimator
style, returning expectation values for a set of observables."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class HybridSelfAttention:
    """
    Quantum self‑attention with a learnable quanvolution front‑end.

    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D filter; the number of qubits is kernel_size**2.
    threshold : float
        Threshold for classical‑to‑quantum data encoding.
    shots : int
        Number of shots for measurement.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 1024):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution filter
        self._filter_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(self.theta):
            self._filter_circuit.rx(p, i)
        self._filter_circuit.barrier()
        self._filter_circuit += random_circuit(self.n_qubits, 2)
        self._filter_circuit.measure_all()

    def _build_attention(self, rotation_params, entangle_params):
        """Attach rotation and entanglement layers to the base attention circuit."""
        circ = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure_all()
        return circ

    def _bind_filter(self, data):
        """Encode a 2‑D array into the filter circuit parameters."""
        param_binds = []
        for dat in data.reshape(1, self.n_qubits):
            bind = {p: (np.pi if val > self.threshold else 0) for p, val in zip(self.theta, dat)}
            param_binds.append(bind)
        return param_binds

    def run(self, rotation_params, entangle_params, data):
        """
        Execute the composite circuit and return measurement counts.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation gates of the attention part.
        entangle_params : np.ndarray
            Parameters for the controlled‑R‑X gates of the attention part.
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) to be encoded.

        Returns
        -------
        dict
            Measurement outcome frequencies.
        """
        results = {}
        for bind in self._bind_filter(data):
            filtered = self._filter_circuit.assign_parameters(bind, inplace=False)
            attn = self._build_attention(rotation_params, entangle_params)
            full = filtered.compose(attn)
            job = qiskit.execute(full, self.backend, shots=self.shots)
            res = job.result().get_counts(full)
            for k, v in res.items():
                results[k] = results.get(k, 0) + v
        return results

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Evaluate expectation values for each parameter set.  Mimics
        the FastBaseEstimator interface used in the classical counterpart.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            rotation_params, entangle_params = params
            # Build a dummy circuit for expectation evaluation
            circ = self._build_attention(rotation_params, entangle_params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["HybridSelfAttention"]
