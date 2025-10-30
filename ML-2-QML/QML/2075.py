import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit import Aer

class FullyConnectedLayer:
    """
    Quantum fully‑connected layer implemented with a parameterised Ry circuit
    and optional entanglement.

    Parameters
    ----------
    n_qubits : int
        Number of qubits that encode the input feature vector.
    entanglement : str
        Pattern of CX gates: 'none', 'full', or 'nearest'.
    backend : qiskit.providers.Backend | None
        Execution backend.  If None, the Aer qasm simulator is used.
    shots : int
        Number of shots for expectation estimation.
    """

    def __init__(self,
                 n_qubits: int,
                 entanglement: str = "full",
                 backend: Backend | None = None,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vector for Ry gates
        self.params = ParameterVector("theta", length=n_qubits)

        # Build the circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for i in range(n_qubits):
            self.circuit.ry(self.params[i], i)

        if entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    self.circuit.cx(i, j)
        elif entanglement == "nearest":
            for i in range(n_qubits - 1):
                self.circuit.cx(i, i + 1)

        self.circuit.measure_all()

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of the first qubit in the Z basis.

        Parameters
        ----------
        thetas : array‑like
            Parameter values for the Ry gates.

        Returns
        -------
        np.ndarray
            A one‑element array containing the expectation value.
        """
        param_bindings = [{self.params[i]: v for i, v in enumerate(thetas)}]
        job = qiskit.execute(self.circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_bindings,
                             backend_options={"max_parallel_threads": 2})
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Convert bitstrings to decimal states
        states = np.array([int(b, 2) for b in counts.keys()], dtype=float)
        probs = np.array(list(counts.values()), dtype=float) / self.shots

        # Expectation of Pauli‑Z on qubit 0: +1 for '0', -1 for '1'
        pauli_z = np.where(states & 1, -1.0, 1.0)
        expectation = np.sum(pauli_z * probs)
        return np.array([expectation])

    def gradient(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Estimate the gradient of the expectation with respect to each parameter
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas : array‑like
            Current parameter values.

        Returns
        -------
        np.ndarray
            Gradient vector with the same length as ``thetas``.
        """
        shift = np.pi / 2
        grads = []
        for i in range(len(thetas)):
            shifted_plus = list(thetas)
            shifted_minus = list(thetas)
            shifted_plus[i] += shift
            shifted_minus[i] -= shift
            f_plus = self.run(shifted_plus)[0]
            f_minus = self.run(shifted_minus)[0]
            grads.append(0.5 * (f_plus - f_minus))
        return np.array(grads)

def FCL():
    """Return a default quantum fully‑connected layer with one qubit."""
    return FullyConnectedLayer(n_qubits=1)
