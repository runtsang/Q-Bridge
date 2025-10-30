"""Hybrid quantum self‑attention circuit that optionally appends an auto‑encoder sub‑circuit
and a fraud‑detection style photonic mapping encoded with single‑qubit rotations."""

import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List
from.FraudDetection import FraudLayerParameters
from.FastBaseEstimator import FastBaseEstimator

class HybridSelfAttention:
    """Hybrid quantum self‑attention that combines a rotation‑entanglement block,
    an optional variational auto‑encoder sub‑circuit, and a fraud‑like
    photonic mapping encoded as rotations."""
    def __init__(self,
                 n_qubits: int,
                 autoencoder_qubits: int | None = None,
                 fraud_params: FraudLayerParameters | None = None):
        self.n_qubits = n_qubits
        self.autoencoder_qubits = autoencoder_qubits
        self.fraud_params = fraud_params
        self.backend = qk.Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circ = QuantumCircuit(qr, cr)

        # Base rotation layer
        for i in range(self.n_qubits):
            circ.rx(0.0, i)
            circ.ry(0.0, i)
            circ.rz(0.0, i)

        # Optional auto‑encoder sub‑circuit
        if self.autoencoder_qubits:
            ae = self._autoencoder_subcircuit(self.autoencoder_qubits)
            circ.append(ae, list(range(self.autoencoder_qubits)))

        # Fraud‑like photonic mapping encoded by rotations
        if self.fraud_params:
            self._apply_fraud_layer(circ, self.fraud_params)

        # Entanglement block
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)

        circ.measure(qr, cr)
        return circ

    def _autoencoder_subcircuit(self, num: int) -> QuantumCircuit:
        qc = QuantumCircuit(num)
        qc.append(RealAmplitudes(num, reps=3), range(num))
        return qc

    def _apply_fraud_layer(self, circ: QuantumCircuit, params: FraudLayerParameters) -> None:
        # Interpret photonic parameters as rotation angles on qubits
        for i in range(min(self.n_qubits, 2)):
            circ.ry(params.bs_theta, i)
            circ.rz(params.bs_phi, i)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circ = self._circuit.copy()
        param_values = np.concatenate([rotation_params, entangle_params])
        param_dict = {p: v for p, v in zip(circ.parameters, param_values)}
        circ = circ.assign_parameters(param_dict, inplace=False)
        job = qk.execute(circ, self.backend, shots=shots)
        return job.result().get_counts(circ)

    def estimator(self,
                  observables: Iterable[BaseOperator],
                  parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        est = FastBaseEstimator(self._circuit)
        return est.evaluate(observables, parameter_sets)

def HybridSelfAttentionFactory(n_qubits: int,
                               autoencoder_qubits: int | None = None,
                               fraud_params: FraudLayerParameters | None = None) -> HybridSelfAttention:
    return HybridSelfAttention(n_qubits, autoencoder_qubits, fraud_params)

__all__ = ["HybridSelfAttention", "HybridSelfAttentionFactory"]
