from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

class UnifiedQuantumModel:
    """
    Quantum‑centric builder that mirrors the classical interface of
    ``UnifiedQuantumModel``.  It constructs a depth‑scaled variational
    circuit, exposes the encoding and variational parameters, and
    provides a method to evaluate the circuit on a backend.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits in the circuit.
        depth : int
            Depth of the layered ansatz.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Build the variational ansatz
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # RX encoding
        for idx, param in enumerate(self.encoding):
            qc.rx(param, idx)
        # Depth‑scaled layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    # ------------------------------------------------------------------
    # Observables: Pauli‑Z on each qubit
    # ------------------------------------------------------------------
    def observables(self) -> List[SparsePauliOp]:
        return [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

    # ------------------------------------------------------------------
    # Evaluate the circuit on a backend
    # ------------------------------------------------------------------
    def run(
        self,
        encoding_vals: List[float],
        weight_vals: List[float],
        backend: str = "qasm_simulator",
    ) -> List[float]:
        """
        Bind the provided parameter values, execute the circuit on the
        specified backend, and return the expectation value of Pauli‑Z on
        each qubit.

        Parameters
        ----------
        encoding_vals : list[float]
            Values to bind to the RX encoding parameters.
        weight_vals : list[float]
            Values to bind to the RY variational parameters.
        backend : str, optional
            Qiskit backend name (default: ``qasm_simulator``).

        Returns
        -------
        List[float]
            Expectation values of Pauli‑Z on each qubit.
        """
        from qiskit import Aer, execute

        bound_qc = self.circuit.bind_parameters(
            {self.encoding[i]: encoding_vals[i] for i in range(self.num_qubits)}
        )
        bound_qc = bound_qc.bind_parameters(
            {self.weights[i]: weight_vals[i] for i in range(self.weights.size())}
        )
        job = execute(bound_qc, Aer.get_backend(backend), shots=1)
        result = job.result()
        counts = result.get_counts()
        total_shots = sum(counts.values())
        exps: List[float] = []
        for qubit in range(self.num_qubits):
            exp = 0.0
            for bitstring, freq in counts.items():
                bit = int(bitstring[self.num_qubits - 1 - qubit])
                exp += freq * (1.0 if bit == 0 else -1.0)
            exps.append(exp / total_shots)
        return exps

__all__ = ["UnifiedQuantumModel"]
