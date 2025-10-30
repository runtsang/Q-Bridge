from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QuantumEstimator

class QuantumEstimatorQNN:
    """
    Stand‑alone variational quantum circuit for regression/classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int = 1, depth: int = 1):
        self.num_qubits = num_qubits
        self.depth = depth
        self.input_params = [Parameter(f"inp_{i}") for i in range(num_qubits)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(num_qubits * depth)]
        self.circuit = self._build_circuit()
        self.observables = SparsePauliOp.from_list([("Y" * num_qubits, 1.0)])
        self.estimator = QuantumEstimator()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a layered ansatz with Ry encoding and Rx rotations."""
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        # Variational layers
        for d in range(self.depth):
            for i, p in enumerate(self.weight_params[d * self.num_qubits:(d + 1) * self.num_qubits]):
                qc.rx(p, i)
            # Entangling
            if self.num_qubits > 1:
                for i in range(self.num_qubits - 1):
                    qc.cx(i, i + 1)
        return qc

    def evaluate(self, inputs: list[float]) -> float:
        """
        Evaluate the circuit on a single input vector.

        Parameters
        ----------
        inputs : list[float]
            Classical feature vector of length *num_qubits*.

        Returns
        -------
        float
            Expectation value of the observable.
        """
        bind_dict = {p: float(v) for p, v in zip(self.input_params, inputs)}
        bound_qc = self.circuit.bind_parameters(bind_dict)
        result = self.estimator.run(
            circuits=[bound_qc],
            observables=[self.observables]
        ).result()
        return result.quasi_results[0].expectation

    def get_weight_params(self):
        """Return the list of trainable weight parameters."""
        return self.weight_params

__all__ = ["QuantumEstimatorQNN"]
