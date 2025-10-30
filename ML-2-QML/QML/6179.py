import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

def build_fraud_detection_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, list[SparsePauliOp]]:
    """
    Construct a quantum feature extractor for fraud detection.
    The circuit encodes a 2‑dimensional input into qubits via RX gates,
    then applies a parameterised variational ansatz.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, encoding, weights, observables

class FraudHybridDetector:
    """
    Quantum encoder that provides expectation values of Z on each qubit.
    The outputs can be used as features for a downstream classical network.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 1, backend: str = "statevector_simulator") -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = Aer.get_backend(backend)
        self.circuit, self.encoding, self.weights, self.observables = build_fraud_detection_circuit(num_qubits, depth)
        self.param_values = {p: 0.0 for p in list(self.encoding) + list(self.weights)}

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Return the expectation values of Z on each qubit for the given input data.
        """
        if data.shape[0]!= self.num_qubits:
            raise ValueError(f"Input must have {self.num_qubits} features.")
        bind = {p: float(v) for p, v in zip(self.encoding, data)}
        bound_circuit = self.circuit.bind_parameters(bind)
        result = execute(bound_circuit, self.backend, shots=1024).result()
        if self.backend.configuration().simulator:
            state = Statevector(result.get_statevector(bound_circuit))
            return np.array([state.expectation_value(obs) for obs in self.observables], dtype=np.float64)
        else:
            counts = result.get_counts(bound_circuit)
            probs = np.array([counts.get(k, 0) for k in result.get_counts(bound_circuit)]) / 1024
            return probs  # placeholder

    def update_weights(self, new_weights: np.ndarray) -> None:
        """
        Assign new variational parameters to the circuit.
        """
        if new_weights.size!= len(self.weights):
            raise ValueError("Incorrect weight vector size.")
        for p, val in zip(self.weights, new_weights):
            self.param_values[p] = float(val)

    def get_parameters(self) -> np.ndarray:
        return np.array([self.param_values[p] for p in self.weights], dtype=np.float64)

    def set_parameters(self, params: np.ndarray) -> None:
        self.update_weights(params)

__all__ = ["build_fraud_detection_circuit", "FraudHybridDetector"]
