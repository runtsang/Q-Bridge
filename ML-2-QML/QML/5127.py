import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter, ParameterVector

class UnifiedQuantumClassicLayer:
    """
    Quantum counterpart of UnifiedQuantumClassicLayer.
    Provides parameterised circuits for FC, SA, QCNN, and AE modes.
    """

    def __init__(self, mode: str = "FC", **kwargs):
        self.mode = mode.upper()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = kwargs.get("shots", 1024)

        if self.mode == "FC":
            self.n_qubits = kwargs.get("n_qubits", 1)
            self.circuit = self._fc_circuit()

        elif self.mode == "SA":
            self.n_qubits = kwargs.get("n_qubits", 4)
            self.circuit = self._sa_circuit()

        elif self.mode == "CNN":
            self.n_qubits = kwargs.get("n_qubits", 8)
            self.circuit = self._cnn_circuit()

        elif self.mode == "AE":
            self.n_qubits = kwargs.get("n_qubits", 5)
            self.circuit = self._ae_circuit()

        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _fc_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        theta = Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def _sa_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        rot = ParameterVector("rot", length=3 * self.n_qubits)
        ent = ParameterVector("ent", length=self.n_qubits - 1)
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.measure_all()
        return qc

    def _cnn_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Convolution: simple pairwise CX + Ry
        for i in range(0, self.n_qubits - 1, 2):
            qc.cx(i, i + 1)
            qc.ry(np.pi / 4, i)
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def _ae_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, 1)
        # Encode: Hadamard on all but the last qubit
        for i in range(self.n_qubits - 1):
            qc.h(i)
        # Swap test
        qc.h(self.n_qubits - 1)
        for i in range(self.n_qubits - 1):
            qc.cswap(self.n_qubits - 1, i, i)
        qc.h(self.n_qubits - 1)
        qc.measure(self.n_qubits - 1, 0)
        return qc

    def run(self, params: dict) -> np.ndarray:
        """
        Execute the circuit with the supplied parameter bindings.
        For FC: {'theta': value}
        For SA: {'rot': array, 'ent': array}
        For CNN/AE: params can be empty.
        """
        if self.mode == "FC":
            bound = self.circuit.bind_parameters({"theta": params.get("theta", 0.0)})
        elif self.mode == "SA":
            bound = self.circuit.bind_parameters({
                "rot": params.get("rot", np.zeros(3 * self.n_qubits)),
                "ent": params.get("ent", np.zeros(self.n_qubits - 1))
            })
        else:
            bound = self.circuit

        job = execute(bound, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound)

        # Map counts to a simple expectation value
        if self.mode == "AE":
            prob_one = counts.get("1", 0) / self.shots
            return np.array([prob_one])
        else:
            prob_zero = counts.get("0", 0) / self.shots
            return np.array([prob_zero])

__all__ = ["UnifiedQuantumClassicLayer"]
