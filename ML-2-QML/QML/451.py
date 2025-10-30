import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp, StateFn

class SelfAttention:
    """
    Variational self‑attention circuit that maps input amplitudes to a
    probability distribution over computational basis states.
    The output is the expectation of Pauli‑Z on each qubit weighted
    by the input values, mimicking a classical attention score.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("aer_simulator")

    def _build_circuit(self, rotation_params, entangle_params, inputs):
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            # encode input as rotation around Y
            circuit.ry(inputs[i], i)
            # apply rotation params
            circuit.rx(rotation_params[3*i], i)
            circuit.ry(rotation_params[3*i+1], i)
            circuit.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            circuit.crx(entangle_params[i], i, i+1)
        return circuit

    def run(self, backend, rotation_params, entangle_params, inputs, shots=1024):
        """
        Parameters
        ----------
        backend : qiskit.providers.BaseBackend
            Quantum backend to execute the circuit on.
        rotation_params : np.ndarray
            Array of shape (3*n_qubits,) controlling single‑qubit rotations.
        entangle_params : np.ndarray
            Array of shape (n_qubits-1,) controlling controlled rotations.
        inputs : np.ndarray
            One‑dimensional array of length n_qubits with input values.
        shots : int, default 1024
            Number of measurement shots.
        Returns
        -------
        np.ndarray
            Output vector of shape (n_qubits,) that mimics a weighted
            attention score.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()

        if isinstance(backend, Aer.AerSimulator):
            state = Statevector(result.get_statevector(circuit))
            exp_z = []
            for i in range(self.n_qubits):
                op = PauliSumOp.from_list([("Z" + "I" * (self.n_qubits-1-i), 1)])
                exp = StateFn(op, is_measurement=True).bind_parameters({}).eval(state)
                exp_z.append(np.real(exp))
            output = np.array(exp_z) * inputs
            return output
        else:
            counts = result.get_counts(circuit)
            total = sum(counts.values())
            probs = np.zeros(self.n_qubits)
            for state, count in counts.items():
                prob = count / total
                bits = np.array(list(reversed(state)), dtype=int)
                probs += prob * bits
            output = probs * inputs
            return output

def SelfAttention():
    """
    Factory that returns a ready‑to‑use SelfAttention instance.
    """
    return SelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
