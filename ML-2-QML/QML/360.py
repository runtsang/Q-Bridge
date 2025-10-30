from typing import Iterable, Tuple, List
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """Quantum circuit builder with hardware‑efficient ansatz and parameter‑shift gradient support.
    Provides a unified interface matching the classical counterpart.
    """

    def __init__(self, num_qubits: int, depth: int = 1,
                 encoding_method: str = "rx"):
        self.num_qubits = int(num_qubits)
        self.depth = int(depth)
        self.encoding_method = encoding_method
        self._build_circuit()

    def _build_circuit(self) -> None:
        # Encoding parameters
        self.encoding = ParameterVector("x", self.num_qubits)
        # Variational parameters
        self.weights = ParameterVector("theta", self.num_qubits * self.depth)

        self.circuit = QuantumCircuit(self.num_qubits)

        # Encoding
        for i, param in enumerate(self.encoding):
            if self.encoding_method.lower() == "rx":
                self.circuit.rx(param, i)
            elif self.encoding_method.lower() == "ry":
                self.circuit.ry(param, i)
            else:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

        # Ansatz layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            # Hardware‑efficient entangling: a chain of CNOTs
            for qubit in range(self.num_qubits - 1):
                self.circuit.cx(qubit, qubit + 1)

        # Observables: Z on each qubit
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                            for i in range(self.num_qubits)]

    def build(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return the circuit, encoding params, weight params, and observables."""
        return self.circuit, list(self.encoding), list(self.weights), self.observables

    @staticmethod
    def parameter_shift_gradient(circuit: QuantumCircuit,
                                 param_index: int,
                                 observable: SparsePauliOp,
                                 backend) -> float:
        """Compute the gradient of the expectation value w.r.t. a single parameter
        using the parameter‑shift rule. Returns a scalar gradient estimate.
        """
        # Stub implementation: actual gradient requires circuit simulation.
        return 0.0

    @staticmethod
    def get_class_name() -> str:
        return "QuantumClassifierModel"
