import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical interface.
    Adds parameter‑shift gradient evaluation, a ring‑entanglement pattern,
    and an explicit data‑encoding scheme to the original seed.
    """
    def __init__(self, num_qubits: int, depth: int = 2,
                 backend=None, seed: int = None, device: str = "cpu"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("aer_simulator_statevector")
        self.seed = seed
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        # initialise parameters to small random values
        np.random.seed(self.seed)
        self.params = np.random.randn(len(self.weights)) * 0.1

    def _build_circuit(self) -> tuple:
        """
        Construct a layered ansatz with data‑encoding, variational layers
        and a ring of CZ gates.  Returns the circuit, encoding vector,
        weight vector and measurement observables.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)

        # data encoding
        for idx, qubit in enumerate(range(self.num_qubits)):
            qc.rx(encoding[idx], qubit)

        # variational layers
        w_idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[w_idx], qubit)
                w_idx += 1
            # entanglement pattern – ring of CZ
            for qubit in range(self.num_qubits):
                qc.cz(qubit, (qubit + 1) % self.num_qubits)

        # observables: Pauli‑Z on each qubit (binary read‑out)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def _expectation(self, params: np.ndarray, data_x: np.ndarray) -> np.ndarray:
        """
        Evaluate the expectation values of the observables for the
        current parameter set and data point.
        """
        # bind all parameters (encoding + variational)
        param_dict = {p: val for p, val in zip(self.encoding, data_x)}
        param_dict.update({p: val for p, val in zip(self.weights, params)})
        bound_qc = self.circuit.bind_parameters(param_dict)

        # obtain the statevector
        job = execute(bound_qc, self.backend, shots=1024)
        result = job.result()
        state = result.get_statevector(bound_qc)

        # compute expectation values
        exp_vals = []
        for op in self.observables:
            mat = op.to_matrix(state.shape[0]).toarray()
            exp = np.real(np.vdot(state, mat @ state))
            exp_vals.append(exp)
        return np.array(exp_vals)

    def _loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """
        Cross‑entropy loss computed from the probability derived from
        the expectation values (after sigmoid).
        """
        preds = []
        for x in X:
            exp = self._expectation(params, x)
            prob = 1 / (1 + np.exp(-exp.sum()))
            preds.append(prob)
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        y = y.astype(np.float32)
        loss = -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
        return loss

    def _parameter_shift_grad(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the parameters
        using the parameter‑shift rule.
        """
        shift = np.pi / 2
        grads = np.zeros_like(params)
        for idx in range(len(params)):
            shift_vec = np.zeros_like(params)
            shift_vec[idx] = shift
            loss_plus = self._loss(X, y, params + shift_vec)
            loss_minus = self._loss(X, y, params - shift_vec)
            grads[idx] = (loss_plus - loss_minus) / (2 * np.sin(shift))
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30,
            lr: float = 0.01, verbose: bool = True) -> None:
        """
        Train the variational parameters using simple gradient descent.
        """
        for epoch in range(epochs):
            loss = self._loss(X, y, self.params)
            grads = self._parameter_shift_grad(X, y, self.params)
            self.params -= lr * grads
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions (0 or 1) based on the sign of the
        summed expectation values.
        """
        preds = []
        for x in X:
            exp = self._expectation(self.params, x)
            prob = 1 / (1 + np.exp(-exp.sum()))
            preds.append(1 if prob > 0.5 else 0)
        return np.array(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy on the given dataset.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_params(self) -> np.ndarray:
        """
        Return the current variational parameters.
        """
        return self.params.copy()

    def set_params(self, params: np.ndarray) -> None:
        """
        Replace the current parameters with the supplied array.
        """
        self.params = params.copy()

__all__ = ["QuantumClassifierModel"]
