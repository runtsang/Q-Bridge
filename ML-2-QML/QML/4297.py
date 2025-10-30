import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

class HybridQuantumFullyConnected:
    """
    Quantum implementation of a hybrid fully‑connected layer.
    Provides a `run` method that accepts a batch of parameter vectors
    and returns the expectation value of a Pauli‑Z measurement on
    the first qubit for each sample.  The circuit consists of
    parameterized RX, RY, RZ gates on each qubit followed by a
    CNOT chain, mirroring the gate‑sharing strategy used in QLSTM.
    """

    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots

        # Build a parameterized circuit template
        self.qreg = QuantumRegister(n_qubits)
        self.creg = ClassicalRegister(n_qubits)
        self.circuit = QuantumCircuit(self.qreg, self.creg)

        # Create parameters for RX, RY, RZ on each qubit
        self.rx_params = [qiskit.circuit.Parameter(f"rx_{i}") for i in range(n_qubits)]
        self.ry_params = [qiskit.circuit.Parameter(f"ry_{i}") for i in range(n_qubits)]
        self.rz_params = [qiskit.circuit.Parameter(f"rz_{i}") for i in range(n_qubits)]

        for i in range(n_qubits):
            self.circuit.rx(self.rx_params[i], i)
            self.circuit.ry(self.ry_params[i], i)
            self.circuit.rz(self.rz_params[i], i)

        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)

        self.circuit.measure_all()

        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each set of parameters in `params`.
        `params` shape: (batch, n_qubits * 3).  The first n_qubits
        elements are RX angles, the next n_qubits are RY, and the
        final n_qubits are RZ.  Returns a numpy array of shape
        (batch, 1) containing the expectation value of Pauli‑Z on
        the first qubit for each sample.
        """
        batch = params.shape[0]
        expectations = np.zeros(batch)

        for idx in range(batch):
            param_bind = {}
            for i in range(self.n_qubits):
                param_bind[self.rx_params[i]] = params[idx, i]
                param_bind[self.ry_params[i]] = params[idx, self.n_qubits + i]
                param_bind[self.rz_params[i]] = params[idx, 2 * self.n_qubits + i]

            job = qiskit.execute(
                self.circuit,
                backend=self.backend,
                shots=self.shots,
                parameter_binds=[param_bind],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            z_exp = np.sum((1 - 2 * (states & 1)) * probs)
            expectations[idx] = z_exp

        return expectations.reshape(-1, 1)
