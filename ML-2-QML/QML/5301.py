import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator

class HybridQuantumCircuit:
    """
    Quantum circuit that implements:
      * Feature‑map rotation (RY)
      * Convolution layers (2‑qubit block)
      * Self‑attention entanglement (CRX)
      * Pooling layers (2‑qubit block)
      * Photonic‑style parameterised gates (RX/RZ)
    The expectation value of the first qubit is returned as the
    quantum analogue of the classical output.
    """
    def __init__(self, n_qubits: int = 8, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.estimator = Estimator()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Feature map
        f_params = ParameterVector("f", self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(f_params[i], i)
        # Convolution layers
        conv_params = ParameterVector("c", self.n_qubits * 3)
        for i in range(0, self.n_qubits, 2):
            idx = i // 2
            qc = qc.compose(self._conv_unit(conv_params[3*idx:3*idx+3]), [i, i+1])
        # Self‑attention entanglement
        att_params = ParameterVector("a", self.n_qubits - 1)
        for i in range(self.n_qubits - 1):
            qc.crx(att_params[i], i, i+1)
        # Pooling layers
        pool_params = ParameterVector("p", (self.n_qubits // 2) * 3)
        for i in range(0, self.n_qubits, 4):
            idx = i // 4
            qc = qc.compose(self._pool_unit(pool_params[3*idx:3*idx+3]), [i//2, i//2+1])
        # Photonic‑style RX/RZ for each qubit
        phot_params = ParameterVector("ph", self.n_qubits * 2)
        for i in range(self.n_qubits):
            qc.rx(phot_params[2*i], i)
            qc.rz(phot_params[2*i+1], i)
        # Measurement
        cr = ClassicalRegister(self.n_qubits)
        qc.add_register(cr)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def _conv_unit(self, params: np.ndarray) -> QuantumCircuit:
        c = QuantumCircuit(2)
        c.rz(-np.pi/2, 1)
        c.cx(1, 0)
        c.rz(params[0], 0)
        c.ry(params[1], 1)
        c.cx(0, 1)
        c.ry(params[2], 1)
        c.cx(1, 0)
        c.rz(np.pi/2, 0)
        return c

    def _pool_unit(self, params: np.ndarray) -> QuantumCircuit:
        p = QuantumCircuit(2)
        p.rz(-np.pi/2, 1)
        p.cx(1, 0)
        p.rz(params[0], 0)
        p.ry(params[1], 1)
        p.cx(0, 1)
        p.ry(params[2], 1)
        return p

    def run(self, param_dict: dict) -> dict:
        """
        Execute the circuit with the supplied parameter bindings.
        Returns the raw measurement counts.
        """
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_dict]
        )
        return job.result().get_counts(self.circuit)

__all__ = ["HybridQuantumCircuit"]
