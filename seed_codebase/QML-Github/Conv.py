import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

def Conv():
    class QuanvCircuit:
        """Filter circuit used for quanvolution layers."""

        def __init__(self, kernel_size, backend, shots, threshold):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            """Run the quantum circuit on classical data.

            Args:
                data: 2D array with shape (kernel_size, kernel_size).

            Returns:
                float: average probability of measuring |1> across qubits.
            """
            data = np.reshape(data, (1, self.n_qubits))

            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = qiskit.execute(self._circuit,
                                self.backend,
                                shots=self.shots,
                                parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)
        
    backend = qiskit.Aer.get_backend("qasm_simulator")
    filter_size = 2
    circuit = QuanvCircuit(filter_size, backend, shots=100, threshold=127)
    return circuit
