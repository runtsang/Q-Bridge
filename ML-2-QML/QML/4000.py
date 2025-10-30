"""Hybrid quantum self‑attention + photonic fraud‑detection circuit.

The class orchestrates a Qiskit self‑attention block followed by a
Strawberry Fields photonic fraud‑detection program, both executed on a
common backend.  The output is a dictionary containing the measurement
counts of the attention circuit and the final photonic state
probabilities, allowing joint analysis of classical and quantum
features.  The design parallels the classical composite model for
direct comparison."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# Re‑use the original parameter container
from dataclasses import dataclass
from typing import Iterable, Dict

# Import the seed helpers (paths are relative to this file)
from.SelfAttention import SelfAttention
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program


@dataclass
class HybridQParams:
    """Container for quantum parameters."""
    rotation_params: np.ndarray
    entangle_params: np.ndarray
    # Photonic fraud‑detection parameters
    fraud_input_params: FraudLayerParameters
    fraud_layer_params: Iterable[FraudLayerParameters]


class HybridQuantumAttentionFraudDetector:
    """Composite QML model integrating Qiskit attention and SF fraud circuit."""

    def __init__(self, n_qubits: int, params: HybridQParams, backend: qiskit.providers.Backend | None = None):
        # Quantum self‑attention
        self.attention = SelfAttention()
        self.attention.__init__(n_qubits)  # reuse constructor from seed
        # Photonic fraud‑detection program
        self.fraud_program = build_fraud_detection_program(
            params.fraud_input_params,
            params.fraud_layer_params,
        )
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def _run_attention(self, shots: int = 1024) -> Dict[str, int]:
        """Execute the quantum self‑attention circuit."""
        circuit = self.attention._build_circuit(
            self.params.rotation_params,
            self.params.entangle_params,
        )
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def _run_fraud(self, state: np.ndarray) -> np.ndarray:
        """Run the photonic fraud‑detection circuit with a given input state."""
        # Initialise a SF program with the same number of modes as qubits
        prog = sf.Program(len(state))
        with prog.context as q:
            # Inject the classical state as displacement amplitudes
            for i, amp in enumerate(state):
                Dgate(amp, 0) | q[i]
            # Apply the fraud program
            for gate in self.fraud_program.program:
                gate.apply(q)
        eng = sf.Engine("gaussian")
        results = eng.run(prog).state
        return results.expectation_value

    def run(self, shots: int = 1024) -> Dict[str, Dict]:
        """Run the full hybrid circuit and return joint diagnostics."""
        # Run attention
        attn_counts = self._run_attention(shots)
        # Estimate a simple feature vector from counts (e.g., most frequent outcome)
        vec = np.array([int(s, 2) for s in attn_counts.most_common(1)[0][0]])
        # Run fraud circuit
        fraud_out = self._run_fraud(vec)
        return {"attention_counts": attn_counts, "fraud_output": fraud_out}
