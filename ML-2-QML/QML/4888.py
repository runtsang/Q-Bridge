import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp

# ---------- Quanvolution layer ----------
class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size, backend, shots, threshold):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# ---------- Variational classifier ----------
def build_classifier_circuit(num_qubits: int, depth: int):
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

# ---------- Quantum fully‑connected layer ----------
class QuantumFullyConnectedLayer:
    """Simple parameterized quantum circuit for demonstration."""
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

# ---------- Hybrid fraud‑detection ----------
class FraudDetectionHybrid:
    """
    Quantum‑assisted fraud‑detection that first processes the 2×2 input patch
    through a photonic‑style quanvolution circuit, then feeds the resulting
    probability into a parameterised ansatz, and finally applies a
    single‑qubit fully‑connected quantum layer for the final score.
    """
    def __init__(self, shots: int = 100):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Build the constituent blocks
        self.conv = QuanvCircuit(2, self.backend, shots, threshold=127)
        self.classifier_circuit, _, _, _ = build_classifier_circuit(num_qubits=1, depth=2)
        self.fcl = QuantumFullyConnectedLayer(1, self.backend, shots)

    def run(self, data: np.ndarray) -> float:
        """
        Run the full hybrid circuit on a 2×2 numpy array and return a
        probability‑based fraud score between 0 and 1.
        """
        # Step 1: quanvolution
        conv_prob = self.conv.run(data)

        # Step 2: classifier – bind the conv‑output as the first rotation parameter
        param_bind = {self.classifier_circuit.parameters[0]: conv_prob}
        job = qiskit.execute(
            self.classifier_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.classifier_circuit)

        # Compute probability of measuring |1> on the single qubit
        prob_1 = sum(val for bit, val in counts.items() if bit == "1") / self.shots

        # Step 3: fully‑connected quantum layer
        expectation = self.fcl.run([prob_1])[0]
        return expectation
