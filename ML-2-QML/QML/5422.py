import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator
from typing import Iterable, List, Sequence
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class HybridSelfAttentionQuantum:
    """
    Quantum‑centric hybrid self‑attention.  A parameterised Qiskit
    circuit produces a probability distribution over the qubits that
    is interpreted as an attention mask.  The class exposes a ``run``
    method that returns this mask and an ``evaluate`` helper that
    mimics the FastBaseEstimator interface.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = AerSimulator()
        self._parameters = None

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement pattern
        for i in range(self.n_qubits - 1):
            circuit.cnot(i, i + 1)
            circuit.rx(entangle_params[i], i + 1)
        circuit.barrier()
        circuit.measure_all()
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return an attention vector of shape
        (n_qubits,) where each entry is the probability of measuring
        |1⟩ on that qubit.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = self.backend.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = {k: v / shots for k, v in counts.items()}
        # Compute marginal probabilities for each qubit
        mask = np.zeros(self.n_qubits)
        for state, p in probs.items():
            for i, bit in enumerate(reversed(state)):
                mask[i] += int(bit) * p
        return mask

    # Estimator utilities ----------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable across all
        parameter sets.  The circuit is rebuilt for each evaluation.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            rot = np.array(params[: self.n_qubits * 3])
            ent = np.array(params[self.n_qubits * 3 :])
            circuit = self._build_circuit(rot, ent)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# Fraud‑detection program (Strawberry Fields) --------------------------------
def build_fraud_detection_program(input_params, layers):
    """
    Placeholder for a photonic fraud‑detection program.  The function
    mirrors the structure of the original example but keeps the
    implementation minimal for clarity.
    """
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    program = sf.Program(2)
    with program.context as q:
        # input layer
        BSgate(input_params.bs_theta, input_params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(input_params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(input_params.squeeze_r, input_params.squeeze_phi)):
            Sgate(r, phi) | q[i]
        BSgate(input_params.bs_theta, input_params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(input_params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(input_params.displacement_r, input_params.displacement_phi)):
            Dgate(r, phi) | q[i]
        for i, k in enumerate(input_params.kerr):
            Kgate(k) | q[i]
        # subsequent layers
        for layer in layers:
            BSgate(layer.bs_theta, layer.bs_phi) | (q[0], q[1])
            for i, phase in enumerate(layer.phases):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(layer.squeeze_r, layer.squeeze_phi)):
                Sgate(r, phi) | q[i]
            BSgate(layer.bs_theta, layer.bs_phi) | (q[0], q[1])
            for i, phase in enumerate(layer.phases):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(layer.displacement_r, layer.displacement_phi)):
                Dgate(r, phi) | q[i]
            for i, k in enumerate(layer.kerr):
                Kgate(k) | q[i]
    return program

__all__ = ["HybridSelfAttentionQuantum", "build_fraud_detection_program"]
