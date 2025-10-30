"""Hybrid quantum convolutional transformer module.

Implements a parameterized quantum circuit for the convolutional
filter, a quantum self‑attention sub‑circuit, and a simple sampler
for probabilistic output.  The public API mirrors the classical
implementation so that the same ``Conv()`` function can be used
across regimes.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler

class HybridConvTransformer:
    """Quantum hybrid model: quantum conv filter + quantum self‑attention + sampler."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 100,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2
        self._build_conv_circuit()
        self._build_attention_circuit()
        self._build_sampler_circuit()

    def _build_conv_circuit(self):
        self.conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.conv_circuit.rx(self.theta[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(self.n_qubits, 2)
        self.conv_circuit.measure_all()

    def _build_attention_circuit(self):
        self.attn_circuit = QuantumCircuit(self.n_qubits)
        self.attn_rot = [qiskit.circuit.Parameter(f"rot{i}") for i in range(3 * self.n_qubits)]
        self.attn_ent = [qiskit.circuit.Parameter(f"ent{i}") for i in range(self.n_qubits - 1)]
        for i in range(self.n_qubits):
            self.attn_circuit.rx(self.attn_rot[3 * i], i)
            self.attn_circuit.ry(self.attn_rot[3 * i + 1], i)
            self.attn_circuit.rz(self.attn_rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            self.attn_circuit.crx(self.attn_ent[i], i, i + 1)
        self.attn_circuit.measure_all()

    def _build_sampler_circuit(self):
        # Simple 2‑qubit sampler inspired by the QML seed
        self.sampler_circuit = QuantumCircuit(2)
        self.sampler_circuit.ry(0.5, 0)
        self.sampler_circuit.ry(0.5, 1)
        self.sampler_circuit.cx(0, 1)
        self.sampler_circuit.ry(0.3, 0)
        self.sampler_circuit.ry(0.3, 1)
        self.sampler_circuit.cx(0, 1)
        self.sampler_circuit.ry(0.2, 0)
        self.sampler_circuit.ry(0.2, 1)

    def run(self, data: np.ndarray) -> float:
        """Execute the hybrid pipeline on a 2‑D array of shape (kernel_size, kernel_size)."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = execute(self.conv_circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.conv_circuit)
        # Average probability of measuring |1> across all qubits
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        avg_prob = counts / (self.shots * self.n_qubits)

        # Self‑attention step
        rot_params = np.full(3 * self.n_qubits, avg_prob)
        ent_params = np.full(self.n_qubits - 1, avg_prob)
        bind = {p: v for p, v in zip(self.attn_rot, rot_params)}
        bind.update({p: v for p, v in zip(self.attn_ent, ent_params)})
        job2 = execute(self.attn_circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result2 = job2.result().get_counts(self.attn_circuit)
        counts2 = 0
        for key, val in result2.items():
            ones = sum(int(bit) for bit in key)
            counts2 += ones * val
        attn_output = counts2 / (self.shots * self.n_qubits)

        # Sampler step
        sampler = StatevectorSampler()
        state = sampler.run(self.sampler_circuit)
        probs = state.get_probabilities()
        # Weighted combination of sampler probabilities
        return attn_output * probs[1] + (1 - attn_output) * probs[0]

def Conv() -> HybridConvTransformer:
    """Return a hybrid quantum Conv‑Transformer instance."""
    return HybridConvTransformer()
