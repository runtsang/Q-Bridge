"""Quantum sampler that mirrors the classical self‑attention architecture."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit import Aer


class QuantumSelfAttention:
    """Parameterised circuit that emulates a self‑attention style block."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", n_qubits * 3)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Encode inputs
        for i in range(self.n_qubits):
            qc.ry(self.input_params[i], i)
        # Entanglement layer (CRX style)
        for i in range(self.n_qubits - 1):
            qc.crx(self.weight_params[3 * i], i, i + 1)
        # Final rotation layer
        for i in range(self.n_qubits):
            qc.ry(self.weight_params[3 * i + 2], i)
        qc.measure_all()
        return qc

    def run(
        self,
        backend,
        input_values: np.ndarray,
        weight_values: np.ndarray,
        shots: int = 1024,
    ) -> dict[str, int]:
        qc = self._build_circuit()
        qc.bind_parameters(
            {p: v for p, v in zip(self.input_params, input_values)}
            | {p: v for p, v in zip(self.weight_params, weight_values)}
        )
        job = backend.run(qc, shots=shots)
        return job.result().get_counts(qc)


def SamplerAttentionQNN() -> SamplerQNN:
    """Wrap the quantum self‑attention circuit as a QNN sampler."""
    backend = Aer.get_backend("qasm_simulator")
    attention = QuantumSelfAttention(n_qubits=4)
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        circuit=attention._build_circuit(),
        input_params=attention.input_params,
        weight_params=attention.weight_params,
        sampler=sampler,
    )
    return qnn


__all__ = ["SamplerAttentionQNN"]
