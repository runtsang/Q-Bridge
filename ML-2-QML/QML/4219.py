import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable

class QuantumFC:
    """Variational quantum circuit acting as a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[{self.theta: theta} for theta in thetas])
        result = job.result()
        counts = result.get_counts(self._circuit)
        counts_arr = np.array(list(counts.values()))
        states = np.array([int(k, 2) for k in counts.keys()])
        probs = counts_arr / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

class QuantumSelfAttention:
    """Quantum‑enhanced self‑attention block."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i)
        qc.measure(qr, cr)
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        counts_arr = np.array(list(counts.values()))
        states = np.array([int(k, 2) for k in counts.keys()])
        probs = counts_arr / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

class QuantumQLSTMGate:
    """Small quantum‑enhanced LSTM cell."""
    def __init__(self, n_qubits: int = 4, shots: int = 512):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _encode_input(self, x: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)
        for i, val in enumerate(x):
            qc.ry(val, i)
        return qc

    def _entangle_and_measure(self, qc: QuantumCircuit) -> np.ndarray:
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        cr = ClassicalRegister(self.n_qubits, "c")
        qc.add_register(cr)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def forward(self, input_vec: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
        qc = self._encode_input(input_vec)
        for i, val in enumerate(prev_state):
            qc.rx(val, i)
        return self._entangle_and_measure(qc)

class UnifiedQuantumLayer:
    """Quantum‑centric implementation of the hybrid layer."""
    def __init__(self, n_features: int = 1, embed_dim: int = 4, n_qubits: int = 4, shots: int = 1024):
        self.fc = QuantumFC(n_qubits=1, shots=shots)
        self.attention = QuantumSelfAttention(n_qubits=embed_dim, shots=shots)
        self.lstm_gate = QuantumQLSTMGate(n_qubits=n_qubits, shots=shots)

    def run(self, mode: str, *args):
        """
        Dispatch to quantum sub‑modules.

        Parameters
        ----------
        mode : str
            One of ``'fc'``, ``'attention'`` or ``'lstm'``.
        """
        if mode == "fc":
            return self.fc.run(*args)
        elif mode == "attention":
            return self.attention.run(*args)
        elif mode == "lstm":
            return self.lstm_gate.forward(*args)
        else:
            raise ValueError(f"Unknown mode {mode}")
