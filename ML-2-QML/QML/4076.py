"""Quantum self‑attention module with variational circuit and classical head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate

class SelfAttentionGen042Quantum:
    """
    Quantum self‑attention front‑end that encodes classical inputs as rotation angles
    and applies a variational ansatz mimicking a self‑attention layer.
    """
    def __init__(self, n_qubits: int = 4, depth: int = 2, num_classes: int | None = None):
        self.n_qubits = n_qubits
        self.depth = depth
        self.num_classes = num_classes

        # parameters
        self.enc_params = ParameterVector("enc", n_qubits)
        self.var_params = ParameterVector("theta", n_qubits * depth)

        self.circuit = self._build_circuit()

        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

        # classical head
        if num_classes is None:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(n_qubits, num_classes)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # encoding
        for i, p in enumerate(self.enc_params):
            qc.ry(p, i)
        # variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                qc.rx(self.var_params[idx], i)
                idx += 1
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, inputs: np.ndarray, param_values: dict, shots: int | None = None) -> torch.Tensor:
        """
        Execute the circuit for each sample in `inputs`.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, n_qubits). Each entry is a rotation angle in radians.
        param_values : dict
            Mapping of ParameterVector names to arrays of shape (n_qubits) or (n_qubits*depth).
        shots : int, optional
            Number of shots; defaults to 1024.
        """
        bound = self.circuit.bind_parameters(
            {p: param_values[p.name] for p in self.circuit.parameters}
        )
        counts_list = []
        for angles in inputs:
            # amplitude encode angles into a statevector
            init_sv = Statevector.from_label("0" * self.n_qubits)
            for i, a in enumerate(angles):
                rot = UnitaryGate(
                    np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                )
                init_sv = init_sv.evolve(rot, [i])
            job = execute(bound, self.backend, shots=shots or self.shots, initial_state=init_sv)
            result = job.result()
            counts_list.append(result.get_counts(bound))

        exp_vals = []
        for cnt in counts_list:
            total = sum(cnt.values())
            exp = 0.0
            for bitstr, c in cnt.items():
                parity = (-1) ** bitstr.count("1")
                exp += parity * c / total
            exp_vals.append(exp)
        exp_tensor = torch.tensor(exp_vals, dtype=torch.float32)

        return self.head(exp_tensor)

def get_SelfAttentionGen042Quantum() -> type:
    """
    Factory that returns the SelfAttentionGen042Quantum class.
    """
    return SelfAttentionGen042Quantum

__all__ = ["SelfAttentionGen042Quantum", "get_SelfAttentionGen042Quantum"]
