from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import List, Tuple

class QuantumClassifierModel:
    """
    Variational quantum classifier with data‑re‑uploading.
    The circuit consists of an encoding layer followed by `depth` layers of
    single‑qubit rotations and nearest‑neighbour CZ gates.  The class exposes
    the same tuple interface as the classical helper: (circuit, encoding,
    weights, observables).  A lightweight training routine uses the
    parameter‑shift rule and a classical Adam optimiser to minimise a cross‑entropy
    loss on a simulated dataset.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        backend=None
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the circuit and capture metadata
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        # Initialize variational parameters
        self.var_params = np.random.randn(len(self.weights))

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for i, qubit in enumerate(range(self.num_qubits)):
            qc.rx(encoding[i], qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables for each qubit (Z)
        obs = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
               for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), obs

    def expectation_values(self, data_batch: np.ndarray, var_params: np.ndarray) -> np.ndarray:
        """
        Evaluate expectation values of the observables for a batch of classical
        inputs.  `data_batch` has shape (batch, num_qubits) and `var_params`
        contains the variational parameters.
        """
        exp_vals = []
        for x in data_batch:
            param_dict = {p: v for p, v in zip(self.encoding, x)}
            for p, v in zip(self.weights, var_params):
                param_dict[p] = v
            bound_qc = self.circuit.bind_parameters(param_dict)
            job = execute(bound_qc, self.backend, shots=1024)
            result = job.result()
            vals = [result.get_expectation_value(obs, bound_qc) for obs in self.observables]
            exp_vals.append(vals)
        return np.array(exp_vals)

    def _parameter_shift_grad(
        self,
        var_params: np.ndarray,
        data_batch: np.ndarray,
        labels_batch: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradients via the parameter‑shift rule for the variational
        parameters only.
        """
        shift = np.pi / 2
        grads = np.zeros_like(var_params)
        for i in range(len(var_params)):
            plus = var_params.copy()
            minus = var_params.copy()
            plus[i] += shift
            minus[i] -= shift
            loss_plus = cross_entropy_loss(
                self.expectation_values(data_batch, plus), labels_batch
            )
            loss_minus = cross_entropy_loss(
                self.expectation_values(data_batch, minus), labels_batch
            )
            grads[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))
        return grads

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 32
    ):
        """
        Simple training loop using Adam optimiser and parameter‑shift gradients.
        """
        optimizer = AdamOptimizer(lr, len(self.var_params))
        for epoch in range(epochs):
            perm = np.random.permutation(len(data))
            data_shuffled = data[perm]
            labels_shuffled = labels[perm]
            for start in range(0, len(data), batch_size):
                batch_x = data_shuffled[start:start + batch_size]
                batch_y = labels_shuffled[start:start + batch_size]

                loss = cross_entropy_loss(
                    self.expectation_values(batch_x, self.var_params), batch_y
                )
                grads = self._parameter_shift_grad(
                    self.var_params, batch_x, batch_y
                )
                self.var_params = optimizer.update(self.var_params, grads)
            print(f"Epoch {epoch + 1}/{epochs} – loss: {loss:.4f}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of classical inputs.  Each input is first
        encoded into the circuit parameters, then the expectation values are
        interpreted as logits.
        """
        logits = self.expectation_values(data, self.var_params)[:, 0]
        probs = 1 / (1 + np.exp(-logits))
        return (probs > 0.5).astype(int)

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        return list(range(self.num_qubits)), [1] * len(self.weights), [0, 1]

    def __repr__(self):
        return f"<QuantumClassifierModel qubits={self.num_qubits} depth={self.depth}>"

# --- Helper utilities ---------------------------------------------------------

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Binary cross‑entropy loss using the first qubit’s expectation value as logit.
    """
    probs = 1 / (1 + np.exp(-logits[:, 0]))
    eps = 1e-12
    return -np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps))

class AdamOptimizer:
    """
    Minimal Adam optimiser for NumPy arrays.
    """
    def __init__(self, lr: float, param_size: int, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.m = np.zeros(param_size)
        self.v = np.zeros(param_size)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

__all__ = ["QuantumClassifierModel"]
