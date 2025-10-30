from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def build_classifier_circuit(num_qubits: int, depth: int):
    """Create a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)
    for p, q in zip(encoding, range(num_qubits)):
        qc.rx(p, q)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    observables = [SparsePauliOp.from_list([( "I" * i + "Z" + "I" * (num_qubits - i - 1), 1 )]) for i in range(num_qubits)]
    return qc, encoding, weights, observables

class EstimatorQNN:
    """Quantum estimator‑based regression layer."""
    def __init__(self, backend=None, shots=1024):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        self.circuit = qc
        self.params = params
        self.observable = SparsePauliOp.from_list([("Y", 1)])

    def run(self, thetas: np.ndarray) -> np.ndarray:
        bind = {self.params[0]: thetas[0], self.params[1]: thetas[1]}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

class HybridQuantumClassifier:
    """Quantum‑only hybrid that mirrors the classical interface."""
    def __init__(self, num_qubits: int = 2, depth: int = 2, backend=None, shots=1024):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.estimator = EstimatorQNN(backend=self.backend, shots=self.shots)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the variational circuit with input parameters and return logits."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        logits = []
        for inp in inputs:
            bind = {p: float(v) for p, v in zip(self.encoding, inp)}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            expectation = np.sum(states * probs)
            logits.append(expectation)
        return np.array(logits)

    def predict_regression(self, thetas: np.ndarray) -> np.ndarray:
        """Return regression output from the quantum EstimatorQNN."""
        return self.estimator.run(thetas)

__all__ = [
    "build_classifier_circuit",
    "EstimatorQNN",
    "HybridQuantumClassifier",
]
