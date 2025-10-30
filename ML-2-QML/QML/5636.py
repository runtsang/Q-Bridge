"""Quantum neural network for regression based on Qiskit Machine Learning.

The circuit uses a 3‑qubit ansatz with data‑encoding, entangling and
parameter‑shift gradient estimation.  It can be trained with the
``StatevectorEstimator`` or ``SamplerEstimator`` primitives.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

class EstimatorQNN:
    """Quantum neural network wrapper.

    Parameters
    ----------
    input_dim : int
        Number of classical input features.
    qubits : int
        Number of qubits in the ansatz.
    depth : int
        Number of variational layers.
    """

    def __init__(
        self,
        input_dim: int = 2,
        qubits: int = 3,
        depth: int = 2,
    ) -> None:
        self.input_dim = input_dim
        self.qubits = qubits
        self.depth = depth
        self.input_params = [Parameter(f"x{i}") for i in range(input_dim)]
        self.weight_params = [
            Parameter(f"w{d}_{q}") for d in range(depth) for q in range(qubits)
        ]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Create a layered variational circuit with data encoding."""
        qc = QuantumCircuit(self.qubits)
        # Data encoding (angle encoding on each qubit)
        for i, p in enumerate(self.input_params):
            qc.ry(p, i % self.qubits)
        # Variational layers
        for d in range(self.depth):
            for q in range(self.qubits):
                idx = d * self.qubits + q
                qc.ry(self.weight_params[idx], q)
            # Entangling block
            for q in range(self.qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def observable(self) -> SparsePauliOp:
        """Observable for regression (Pauli‑Z on all qubits)."""
        ops = [("Z" * self.qubits, 1.0)]
        return SparsePauliOp.from_list(ops)

    def estimator_qnn(self) -> QiskitEstimatorQNN:
        """Instantiate Qiskit EstimatorQNN with a state‑vector backend."""
        estimator = Estimator()  # default state‑vector estimator
        return QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable(),
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    # ------------------------------------------------------------------
    # Gradient estimation using parameter‑shift rule
    # ------------------------------------------------------------------
    def parameter_shift_grad(
        self, params: dict, shift: float = 0.5
    ) -> dict:
        """Compute gradient of the expectation value w.r.t. weight parameters.

        Parameters
        ----------
        params : dict
            Mapping from Parameter objects to their numerical values.
        shift : float
            Shift amount for the parameter‑shift rule.

        Returns
        -------
        grad : dict
            Gradient dictionary mapping each weight Parameter to its derivative.
        """
        qnn = self.estimator_qnn()
        grad = {}
        for w in self.weight_params:
            # Shift +s
            params_plus = params.copy()
            params_plus[w] = params[w] + shift
            val_plus = qnn.evaluate(params_plus, self.input_params)[0]
            # Shift -s
            params_minus = params.copy()
            params_minus[w] = params[w] - shift
            val_minus = qnn.evaluate(params_minus, self.input_params)[0]
            grad[w] = (val_plus - val_minus) / (2 * shift)
        return grad

    # ------------------------------------------------------------------
    # Simple training loop (stochastic gradient descent)
    # ------------------------------------------------------------------
    def train(
        self,
        data: list[tuple[list[float], float]],
        epochs: int = 10,
        lr: float = 0.1,
    ) -> list[float]:
        """Train the QNN on a list of (input, target) pairs.

        Parameters
        ----------
        data : list
            Each element is a tuple (features, target).
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.

        Returns
        -------
        losses : list
            Training loss per epoch.
        """
        qnn = self.estimator_qnn()
        # initialise weight parameters randomly
        params = {p: 0.0 for p in self.weight_params}
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                # evaluate current prediction
                full_params = {**params, **dict(zip(self.input_params, x))}
                pred = qnn.evaluate(full_params, self.input_params)[0]
                loss = (pred - y) ** 2
                epoch_loss += loss
                grad = self.parameter_shift_grad(params)
                # gradient descent update
                for w in self.weight_params:
                    params[w] -= lr * grad[w]
            epoch_loss /= len(data)
            losses.append(epoch_loss)
        return losses
