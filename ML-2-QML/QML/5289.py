"""Hybrid quantum–classical classifier that fuses data‑encoding, variational ansatz, attention‑style CX chain, random sub‑circuit and a sampler‑like measurement."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit
import qiskit.circuit.random as random_circuit

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a quantum circuit that mirrors the classical build_classifier_circuit.
    The circuit consists of:
        * a data‑encoding layer (RX rotations),
        * a depth‑× variational ansatz (RY + CZ),
        * a CX‑based attention block,
        * a short random sub‑circuit emulating a quanvolution filter,
        * a final measurement in the computational basis.
    """
    # Encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for i in range(num_qubits):
        qc.rx(encoding[i], i)

    # Variational ansatz
    for d in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[d * num_qubits + i], i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # Attention‑style CX chain
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Random sub‑circuit (emulating a quanvolution filter)
    qc += random_circuit.random_circuit(num_qubits, 2)
    qc.barrier()

    # Measurement
    qc.measure_all()

    # Observables (Pauli‑Z on each qubit)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables


class HybridQuantumClassifier:
    """
    Quantum counterpart to the classical HybridQuantumClassifier.
    The run method evaluates the circuit on a backend and returns the
    expectation values of the Z observables.
    """
    def __init__(self, num_qubits: int, depth: int, backend=None) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def run(self, data: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit for the provided data.

        Parameters
        ----------
        data : np.ndarray, shape (num_qubits,)
            Classical feature vector that will be bound to the encoding parameters.
        shots : int, optional
            Number of shots for the backend simulation.

        Returns
        -------
        expectations : np.ndarray, shape (num_qubits,)
            Expectation values of the Z observables.
        """
        # Bind the encoding parameters
        bind_dict = {p: float(val) for p, val in zip(self.encoding, data)}
        bound_circuit = self.circuit.bind_parameters(bind_dict)

        job = qiskit.execute(bound_circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Convert counts to probabilities
        probs = {bitstring: counts[bitstring] / shots for bitstring in counts}

        # Compute expectation values
        expectations = []
        for op in self.observables:
            exp = 0.0
            for bitstring, p in probs.items():
                # Bitstring is in little‑endian order
                bit = bitstring[::-1]
                # Z expectation: +1 for |0>, -1 for |1>
                z = 1 if bit[op.to_label().index("Z")] == "0" else -1
                exp += z * p
            expectations.append(exp)

        return np.array(expectations)


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
