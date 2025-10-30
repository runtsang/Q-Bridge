"""Quantum self‑attention module with a variational estimator head.

The implementation mirrors the classical SelfAttentionHybrid but replaces
the attention and classification layers with Qiskit variational circuits.
It builds a parameterised circuit for each attention head, measures
Z‑expectation values to obtain attention weights, and uses an
EstimatorQNN with a single‑qubit Y observable as a quantum expectation
head for classification or regression.  The class is fully importable
and can be dropped into any hybrid training loop.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit import Aer

class SelfAttentionHybrid:
    """Quantum‑style multi‑head self‑attention with a variational head."""

    def __init__(
        self,
        n_qubits: int = 4,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
        num_heads: int = 1,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.shift = shift
        self.num_heads = num_heads

        # Rotation and entanglement parameters for attention
        self.rotation_params = [Parameter(f"rot_{i}") for i in range(3 * n_qubits)]
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(n_qubits - 1)]

        # Build attention circuit template
        self.attention_template = self._build_attention_circuit(
            self.rotation_params, self.entangle_params
        )

        # Build a single‑qubit estimator for classification/regression
        self._build_estimator()

    def _build_attention_circuit(
        self,
        rotation_params: list[Parameter],
        entangle_params: list[Parameter],
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Apply rotation gates per qubit
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        # Entangle adjacent qubits
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i + 1)
        qc.measure_all()
        return qc

    def _build_estimator(self) -> None:
        circ = QuantumCircuit(1)
        circ.h(0)
        circ.ry(Parameter("input"), 0)
        circ.rx(Parameter("weight"), 0)
        circ.measure_all()
        obs = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = EstimatorQNN(
            circuit=circ,
            observables=obs,
            input_params=[Parameter("input")],
            weight_params=[Parameter("weight")],
            estimator=StatevectorEstimator(),
        )

    def _run_attention(self, inputs: np.ndarray) -> np.ndarray:
        """Run the attention circuit for a batch of inputs and return Z‑expectations."""
        param_binds = []
        for sample in inputs:
            bind = {}
            for i, val in enumerate(sample):
                bind[self.rotation_params[3 * i]] = val
                bind[self.rotation_params[3 * i + 1]] = val
                bind[self.rotation_params[3 * i + 2]] = val
            for i in range(self.n_qubits - 1):
                bind[self.entangle_params[i]] = 0.0
            param_binds.append(bind)

        compiled = transpile(self.attention_template, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()

        expectations = []
        for counts in result.get_counts():
            # Compute expectation of Z for the entire state
            exp_z = 0.0
            total = sum(counts.values())
            for state, cnt in counts.items():
                z = 1
                for bit in state:
                    z *= 1 if bit == "0" else -1
                exp_z += z * cnt / total
            expectations.append(exp_z)
        return np.array(expectations)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, n_qubits) – each row encodes a feature vector.

        Returns
        -------
        np.ndarray
            Probabilities in [0, 1] for each sample.
        """
        # Attention weights from quantum circuit
        attn_weights = self._run_attention(inputs)
        # Weighted sum of the input features
        weighted = inputs * attn_weights[:, None]
        weighted_sum = weighted.sum(axis=1)
        # Run the estimator on the weighted sum
        preds = []
        for w in weighted_sum:
            out = self.estimator.predict([w])[0]
            preds.append(out)
        preds = np.array(preds)
        # Map Y‑expectation [-1,1] to probability [0,1]
        probs = 0.5 * (preds + 1.0)
        return probs

    def estimate(self, inputs: np.ndarray) -> np.ndarray:
        """Return a scalar regression estimate per sample."""
        attn_weights = self._run_attention(inputs)
        weighted = inputs * attn_weights[:, None]
        weighted_sum = weighted.sum(axis=1)
        preds = []
        for w in weighted_sum:
            preds.append(self.estimator.predict([w])[0])
        return np.array(preds)

__all__ = ["SelfAttentionHybrid"]
