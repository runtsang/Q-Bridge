import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.estimators import SamplerEstimator
from qiskit.quantum_info import SparsePauliOp


class EstimatorQNN:
    """
    Two‑qubit variational neural network for regression.
    • Parameterised RX, RZ rotations and CX entanglement.
    • Uses Pauli‑Z as the observable.
    • Exposes ``predict`` that accepts a NumPy array of shape (n_samples, 2).
    """
    def __init__(self, backend=None, shots: int = 1024):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._build_ansatz()
        self.estimator = SamplerEstimator(backend=self.backend, shots=self.shots)
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=[self.observable],
            input_params=[self.input_params],
            weight_params=[self.weight_params],
            estimator=self.estimator,
        )

    def _build_ansatz(self):
        # Input parameters
        self.input_params = [Parameter("x0"), Parameter("x1")]
        # Variational parameters
        self.weight_params = [Parameter(f"w{i}") for i in range(6)]
        self.circuit = QuantumCircuit(2)
        # Input encoding
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        # Variational layers
        for w in self.weight_params:
            self.circuit.ry(w, 0)
            self.circuit.rz(w, 1)
            self.circuit.cx(0, 1)
        # Observable
        self.observable = SparsePauliOp.from_list([("Z" * 2, 1)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim!= 2 or X.shape[1]!= 2:
            raise ValueError("Input must be (n_samples, 2)")
        preds = []
        # Initialise random weights once
        if not hasattr(self, "weights"):
            self.weights = np.random.uniform(0, 2 * np.pi, len(self.weight_params))
        for x in X:
            param_dict = {
                self.input_params[0]: x[0],
                self.input_params[1]: x[1],
            }
            for w, val in zip(self.weight_params, self.weights):
                param_dict[w] = val
            res = self.qnn.predict(param_dict)
            preds.append(res[0])
        return np.array(preds)


__all__ = ["EstimatorQNN"]
