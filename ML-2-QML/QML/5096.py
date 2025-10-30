import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.quantum_info import Statevector
from typing import Iterable, Sequence

class QuantumFCL:
    """
    Parameterized single‑qubit circuit that emulates a fully‑connected layer.
    """
    def __init__(self, shots: int = 1024):
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys())).astype(float)
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class QuantumKernel:
    """
    Fixed‑parameter quantum kernel based on a 4‑qubit Ry ansatz.
    """
    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.backend = Aer.get_backend('statevector_simulator')

    def _encode(self, x: np.ndarray) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.n_wires)
        for i, val in enumerate(x):
            circ.ry(val, i)
        return circ

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        circ_x = self._encode(x)
        circ_y = self._encode(-y)          # negative encoding as in the reference
        sv_x = Statevector.from_instruction(circ_x)
        sv_y = Statevector.from_instruction(circ_y)
        overlap = np.abs(np.vdot(sv_x.data, sv_y.data)) ** 2
        return overlap

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """
    Evaluate the Gram matrix between datasets ``a`` and ``b`` using the quantum kernel.
    """
    kernel = QuantumKernel()
    mat = np.zeros((len(a), len(b)))
    for i, va in enumerate(a):
        for j, vb in enumerate(b):
            mat[i, j] = kernel.kernel(va, vb)
    return mat

__all__ = ["QuantumFCL", "QuantumKernel", "kernel_matrix"]
