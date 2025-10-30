from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class QuantumClassifier:
    """
    Quantum variational circuit that implements a data‑uploading classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    entanglement : str, optional
        Entanglement pattern: 'full' or 'nearest'. Default is 'full'.
    measurement : str, optional
        Measurement type:'single' (Z on each qubit) or'multi' (multi‑qubit Pauli). Default'single'.
    """

    def __init__(self, num_qubits: int, depth: int,
                 entanglement: str = "full",
                 measurement: str = "single"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.measurement = measurement
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self):
        # Data encoding
        encoding = ParameterVector("x", self.num_qubits)
        # Variational parameters
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            # Entanglement
            if self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qc.cx(i, j)
            elif self.entanglement == "nearest":
                for i in range(self.num_qubits - 1):
                    qc.cx(i, i + 1)

        # Observables
        if self.measurement == "single":
            observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                           for i in range(self.num_qubits)]
        else:  # multi‑qubit Pauli
            observables = [SparsePauliOp("Z" * self.num_qubits)]  # single multi‑qubit observable

        return qc, list(encoding), list(weights), observables

    def get_expectation(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate expectation values for a single data point.

        Parameters
        ----------
        data : array_like, shape (num_qubits,)
            Feature vector to encode.
        params : array_like, shape (num_qubits * depth,)
            Variational parameters.

        Returns
        -------
        expectation : ndarray
            Expectation values of the observables.
        """
        if data.shape[0]!= self.num_qubits:
            raise ValueError("Data dimension must match number of qubits.")
        if params.shape[0]!= len(self.weights):
            raise ValueError("Parameter vector length mismatch.")

        params_dict = {param: val for param, val in zip(self.encoding, data)}
        params_dict.update({param: val for param, val in zip(self.weights, params)})
        bound = self.circuit.bind_parameters(params_dict)

        backend = Aer.get_backend("statevector_simulator")
        job = execute(bound, backend)
        result = job.result()
        state = result.get_statevector(bound)
        expectation = np.array([obs.expectation_value(state).real for obs in self.observables])
        return expectation

__all__ = ["QuantumClassifier"]
