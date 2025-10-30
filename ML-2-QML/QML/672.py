"""ConvEnhancedQML: quantum parameter‑shift convolutional filter."""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer, execute

def ConvEnhancedQML(kernel_size: int = 3,
                    depth: int = 2,
                    shots: int = 1024,
                    threshold: float = 0.5,
                    backend_name: str = "qasm_simulator") -> object:
    """
    Return a quantum circuit object that emulates a convolutional filter.
    The circuit uses a RealAmplitudes ansatz with a given depth and a
    parameter‑shift measurement scheme.  The function accepts 2‑D
    data of shape (kernel_size, kernel_size) and returns the average
    probability of measuring |1> across all qubits.

    Parameters
    ----------
    kernel_size : int
        Size of the filter (number of qubits = kernel_size**2).
    depth : int
        Depth of the RealAmplitudes ansatz.
    shots : int
        Number of shots per execution.
    threshold : float
        Threshold on the classical input values that determines the
        rotation angles (0.0–1.0).
    backend_name : str
        Name of the Qiskit backend to use.

    Returns
    -------
    object
        An object with a ``run`` method that accepts a 2‑D numpy array.
    """
    n_qubits = kernel_size ** 2
    backend = Aer.get_backend(backend_name)

    # Build parameter‑shift ansatz
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=depth, entanglement='full')
    params = [Parameter(f"θ{i}") for i in range(n_qubits)]

    # Build full circuit
    circuit = qiskit.QuantumCircuit(n_qubits)
    circuit.append(ansatz, range(n_qubits))
    circuit.measure_all()

    class QuanvCircuit:
        def __init__(self):
            self.circuit = circuit
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            data = np.asarray(data).reshape(1, n_qubits)
            param_binds = []
            for row in data:
                bind = {}
                for i, val in enumerate(row):
                    angle = np.pi if val > self.threshold else 0.0
                    bind[params[i]] = angle
                param_binds.append(bind)

            job = execute(self.circuit,
                          backend=self.backend,
                          parameter_binds=param_binds,
                          shots=self.shots)
            result = job.result()
            counts = result.get_counts(self.circuit)

            total_ones = 0
            total_counts = 0
            for bitstring, freq in counts.items():
                ones = bitstring.count('1')
                total_ones += ones * freq
                total_counts += freq

            avg_prob = total_ones / (total_counts * n_qubits)
            return avg_prob

    return QuanvCircuit()

__all__ = ["ConvEnhancedQML"]
