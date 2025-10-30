import numpy as np
from qiskit import Aer, execute, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridClassifier:
    """Quantum implementation of the hybrid classifier.

    The circuit follows the layered ansatz from *QuantumClassifierModel.py*,
    with explicit data encoding (RX) followed by a depth‑controlled sequence
    of variational rotations (RY) and entangling CZ gates.  The observable set
    mirrors the classical network: a single Pauli‑Z measurement per qubit.
    """
    def __init__(self, n_qubits: int = 1, depth: int = 1, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self):
        encoding = ParameterVector("x", self.n_qubits)
        weights = ParameterVector("theta", self.n_qubits * self.depth)

        qc = QuantumCircuit(self.n_qubits)
        for param, qubit in zip(encoding, range(self.n_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.n_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Pauli‑Z observables per qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
                       for i in range(self.n_qubits)]
        return qc, list(encoding), list(weights), observables

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the quantum circuit with the supplied parameters.

        Parameters
        ----------
        thetas : np.ndarray
            Flat array of shape ``n_qubits + n_qubits*depth`` containing
            encoding and variational parameters in that order.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value of the first Pauli‑Z
            observable (mirroring the classical single‑output).
        """
        thetas = np.asarray(thetas, dtype=np.float32).flatten()
        if len(thetas)!= len(self.encoding) + len(self.weights):
            raise ValueError("Parameter vector length does not match circuit size")

        param_binds = [
            {p: t for p, t in zip(self.encoding, thetas[:self.n_qubits])},
            {p: t for p, t in zip(self.weights, thetas[self.n_qubits:])}
        ]

        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

def FCL(n_qubits: int = 1, depth: int = 1, mode: str = "quantum",
        backend=None, shots: int = 100) -> HybridClassifier:
    """Factory returning a hybrid classifier in quantum mode.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the variational circuit.
    depth : int
        Depth of the layered ansatz.
    mode : str
        Must be ``"quantum"``; the classical variant is provided in the ML module.
    backend : qiskit backend
        Optional custom backend; defaults to Aer qasm simulator.
    shots : int
        Number of shots for expectation estimation.

    Returns
    -------
    HybridClassifier
        Instance configured for quantum execution.
    """
    if mode!= "quantum":
        raise ValueError("Only quantum mode is supported in the QML module.")
    return HybridClassifier(n_qubits, depth, backend, shots)

__all__ = ["HybridClassifier", "FCL"]
