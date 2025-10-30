import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import QuantumInstance


class QuantumClassifier:
    """
    Quantum implementation of a data‑encoded variational classifier.
    The circuit consists of an RX data‑encoding layer followed by a stack
    of Ry rotations and CZ entanglement.  The class provides train, predict,
    and evaluate using expectation values of Z observables computed on a
    backend (default state‑vector simulator).  Metadata (encoding,
    weight_sizes, observables) mirrors the classical counterpart for
    API compatibility.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 backend_name: str = "statevector_simulator",
                 shots: int = 1024,
                 optimizer_name: str = "SPSA",
                 maxiter: int = 200,
                 lr: float = 0.01):
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("theta", num_qubits * depth)
        self.circuit = self._build_circuit()
        self.encoding = list(range(num_qubits))
        self.weight_sizes = [num_qubits * depth]
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        self.backend = Aer.get_backend(backend_name)
        self.shots = shots
        self.quantum_instance = QuantumInstance(self.backend, shots=shots)
        self.optimizer_name = optimizer_name
        self.maxiter = maxiter
        self.lr = lr
        self.weight_values = np.random.uniform(-np.pi, np.pi,
                                               size=self.weight_params.size())

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # data encoding
        for i, param in enumerate(self.encoding_params):
            qc.rx(param, i)
        # variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weight_params[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def _sample_expectation(self,
                            feature_vector: np.ndarray,
                            weight_params: np.ndarray) -> float:
        param_dict = {
            self.encoding_params[i]: feature_vector[i]
            for i in range(self.num_qubits)
        }
        param_dict.update({
            self.weight_params[i]: weight_params[i]
            for i in range(len(weight_params))
        })
        bound_circ = self.circuit.bind_parameters(param_dict)
        result = execute(bound_circ, self.backend,
                         shots=self.shots).result()
        counts = result.get_counts(bound_circ)
        # expectation of Z on first qubit
        exp = 0.0
        for bitstring, cnt in counts.items():
            z = 1 if bitstring[-1] == "0" else -1
            exp += z * cnt
        exp /= self.shots
        return exp

    def _batch_loss(self,
                    weight_params: np.ndarray,
                    X: np.ndarray,
                    y: np.ndarray) -> float:
        probs = []
        for x_vec, label in zip(X, y):
            exp = self._sample_expectation(x_vec, weight_params)
            p = (exp + 1.0) / 2.0  # map to [0,1]
            probs.append(p)
        probs = np.array(probs)
        eps = 1e-9
        loss = -np.mean(
            y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)
        )
        return loss

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              max_iter: int = None,
              lr: float = None) -> None:
        """
        Optimize the variational parameters using a gradient‑free optimiser.
        """
        if max_iter is None:
            max_iter = self.maxiter
        if lr is None:
            lr = self.lr
        optimizer = SPSA(maxiter=max_iter, step_size=lr, perturbation=0.01)
        init_params = self.weight_values.copy()

        def objective(p):
            return self._batch_loss(p, X, y)

        opt_params, min_val, _ = optimizer.optimize(
            num_vars=len(init_params),
            objective_function=objective,
            initial_point=init_params,
        )
        self.weight_values = opt_params

    def predict(self,
                X: np.ndarray,
                batch_size: int = 64) -> np.ndarray:
        """
        Return class probabilities for the input data.
        """
        probs = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            batch_probs = []
            for x_vec in batch:
                exp = self._sample_expectation(x_vec, self.weight_values)
                p = (exp + 1.0) / 2.0
                batch_probs.append(p)
            probs.append(batch_probs)
        return np.concatenate(probs, axis=0)

    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 64) -> float:
        """
        Return accuracy on the given data.
        """
        probs = self.predict(X, batch_size)
        preds = (probs > 0.5).astype(int)
        return np.mean(preds == y)

    def save(self, path: str) -> None:
        """
        Persist the variational parameters and metadata to disk.
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "weight_values": self.weight_values,
                    "encoding_params": [float(v) for v in self.encoding_params],
                    "weight_params": [float(v) for v in self.weight_params],
                    "encoding": self.encoding,
                    "weight_sizes": self.weight_sizes,
                    "observables": [str(o) for o in self.observables],
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "QuantumClassifier":
        """
        Load a previously saved quantum classifier.
        """
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(num_qubits=len(data["encoding"]),
                  depth=len(data["weight_sizes"]))
        obj.weight_values = data["weight_values"]
        obj.encoding = data["encoding"]
        obj.weight_sizes = data["weight_sizes"]
        obj.observables = data["observables"]
        return obj
