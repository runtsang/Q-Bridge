"""QuantumClassifierModel: quantum‑centric implementation that mirrors
the classical interface defined in the ML module.  It uses Qiskit for
state preparation, a variational Ansatz, and a quantum kernel that
can be employed in SVM‑style training.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# Quanvolution filter
# --------------------------------------------------------------------------- #
class ConvFilterQuantum:
    """Quantum quanvolution filter that emulates the classical Conv filter."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quanvolution circuit on a 2‑D data patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self._circuit, self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """Quantum kernel built from a shallow variational Ansatz."""
    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = Aer.get_backend("statevector_simulator")
        self.circuit = self._build_ansatz()

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(0.01, q)  # placeholder angle
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)
        return qc

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the kernel value as squared overlap of prepared states."""
        x_state = self._prepare_statevector(x)
        y_state = self._prepare_statevector(y)
        return abs(np.vdot(x_state, y_state)) ** 2

    def _prepare_statevector(self, data: np.ndarray) -> np.ndarray:
        qc = self.circuit.copy()
        for q, val in enumerate(data):
            qc.ry(val, q)
        state = Statevector.from_instruction(qc)
        return state.data

def kernel_matrix(a: List[np.ndarray],
                  b: List[np.ndarray],
                  num_qubits: int,
                  depth: int) -> np.ndarray:
    qkernel = QuantumKernel(num_qubits, depth)
    return np.array([[qkernel.evaluate(x, y) for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Quantum circuit factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                  Iterable,
                                                  Iterable,
                                                  List[SparsePauliOp]]:
    """Construct a variational classification circuit and return metadata."""
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Composite quantum classifier
# --------------------------------------------------------------------------- #
class QuantumClassifierModel:
    """
    Quantum drop‑in replacement for the classical model.  The class
    implements a variational circuit, a quanvolution filter, and a
    quantum kernel that can be used for SVM‑style training.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 shots: int = 100):
        self.num_qubits = num_qubits
        self.depth = depth
        self.conv = ConvFilterQuantum(conv_kernel_size,
                                      qiskit.Aer.get_backend("qasm_simulator"),
                                      shots,
                                      conv_threshold)

        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_classifier_circuit()

    def _build_classifier_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Encoding of classical data
        for q in range(self.num_qubits):
            qc.rx(qiskit.circuit.Parameter(f"x{q}"), q)
        # Variational layers
        for d in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(qiskit.circuit.Parameter(f"theta{d}_{q}"), q)
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)
        qc.measure_all()
        return qc

    def predict(self, data: np.ndarray) -> int:
        """Return a binary label by running the circuit and thresholding."""
        # First run the quanvolution filter
        conv_out = self.conv.run(data)

        # Bind encoding parameters from data
        param_binds = {f"x{q}": val for q, val in enumerate(data.flatten()[:self.num_qubits])}
        job = execute(self.circuit,
                      self.backend,
                      shots=1000,
                      parameter_binds=[param_binds])
        result = job.result().get_counts(self.circuit)

        # Compute mean probability of measuring |1>
        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        prob = total / (1000 * self.num_qubits)

        # Simple threshold to give binary output
        return int(prob > 0.5)

    def evaluate_kernel(self,
                        a: List[np.ndarray],
                        b: List[np.ndarray]) -> np.ndarray:
        """Return the quantum kernel Gram matrix."""
        return kernel_matrix(a, b, self.num_qubits, self.depth)

__all__ = [
    "ConvFilterQuantum",
    "QuantumKernel",
    "kernel_matrix",
    "build_classifier_circuit",
    "QuantumClassifierModel",
]
