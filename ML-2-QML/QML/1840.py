from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import List

class QuantumClassifierModel:
    """
    Quantum circuit factory for a variational classifier.

    The circuit implements a hardware‑efficient ansatz with data re‑uploading
    and entangling layers.  It exposes:
        - build_circuit: constructs a parameterized circuit.
        - compute_expectations: evaluates expectation values of a set of
          Pauli observables on a statevector or simulator.
    """
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = Aer.get_backend("statevector_simulator")

    def build_circuit(
        self,
        data_params: List[float],
        weight_params: List[float],
    ) -> QuantumCircuit:
        """
        Build a variational circuit with data re‑uploading.

        Args:
            data_params: List of length num_qubits containing feature values.
            weight_params: List of variational parameters of length
                           num_qubits * depth.

        Returns:
            A QuantumCircuit ready for simulation.
        """
        qc = QuantumCircuit(self.num_qubits)

        # Data encoding using RX rotations
        for qubit, val in enumerate(data_params):
            qc.rx(val, qubit)

        idx = 0
        for _ in range(self.depth):
            # Parameterized Ry gates
            for qubit in range(self.num_qubits):
                qc.ry(weight_params[idx], qubit)
                idx += 1
            # Entangling layer: CX in a ring topology
            for qubit in range(self.num_qubits):
                qc.cx(qubit, (qubit + 1) % self.num_qubits)

        return qc

    def observables(self) -> List[SparsePauliOp]:
        """
        Return a list of Pauli‑Z measurements on each qubit.

        Returns:
            List of SparsePauliOp objects.
        """
        return [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

    def compute_expectations(
        self,
        qc: QuantumCircuit,
        observables: List[SparsePauliOp],
    ) -> List[float]:
        """
        Compute expectation values of given observables on the circuit's state.

        Args:
            qc: QuantumCircuit to evaluate.
            observables: List of SparsePauliOp to measure.

        Returns:
            List of expectation values.
        """
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        statevector = result.get_statevector(qc)
        exp_vals = []
        for op in observables:
            exp_vals.append(op.expectation_value(statevector))
        return exp_vals

__all__ = ["QuantumClassifierModel"]
