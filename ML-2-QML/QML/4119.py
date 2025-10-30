import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridSelfAttentionClassifier:
    """
    Quantum analogue of the classical HybridSelfAttentionClassifier.
    Builds a variational circuit with an encoding layer, an attention‑style
    entanglement block, and a classification ansatz.  The circuit can be
    executed on any Qiskit backend.
    """
    def __init__(self, n_qubits: int = 4, depth: int = 2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        encoding = ParameterVector("x", self.n_qubits)
        for i, param in enumerate(encoding):
            circuit.rx(param, i)

        # Attention block
        attn_params = ParameterVector("theta_attn", self.n_qubits * self.depth)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                circuit.ry(attn_params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                circuit.cz(q, q + 1)

        # Classification ansatz
        cls_params = ParameterVector("theta_cls", self.n_qubits * self.depth)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                circuit.ry(cls_params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                circuit.cz(q, q + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, shots: int = 1024, **param_bindings):
        """
        Execute the circuit on the supplied backend.
        :param backend: Qiskit backend instance.
        :param shots: Number of measurement shots.
        :param param_bindings: Mapping from ParameterVector names to numpy arrays.
        :return: Measurement counts as a dict.
        """
        bound_circ = self.circuit.bind_parameters(param_bindings)
        job = qiskit.execute(bound_circ, backend, shots=shots)
        return job.result().get_counts(bound_circ)

    def observables(self):
        """PauliZ observables on each qubit, analogous to the classical head."""
        return [SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
                for i in range(self.n_qubits)]

__all__ = ["HybridSelfAttentionClassifier"]
