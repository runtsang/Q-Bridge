import torch
from torch import nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

class HybridEstimatorQNN(nn.Module):
    """
    Hybrid regressor that replaces the classical quantum layer with a
    Qiskit‑based variational circuit.  The circuit receives the output
    of the second classical hidden layer as a set of input parameters
    and produces a single measurement that is fed into the final
    linear head.
    """

    def __init__(self, input_dim: int = 2, hidden_sizes: tuple[int, int] = (8, 4),
                 n_qubits: int = 4, depth: int = 2, estimator: QiskitEstimator | None = None):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], 1)

        self.n_qubits = n_qubits
        self.depth = depth
        self._build_qc()

        self.estimator = estimator or QiskitEstimator(method="statevector")
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_qc(self):
        """Create a small circuit that applies a Hadamard, Ry/ Rx layers and a CNOT chain."""
        self.qc = QuantumCircuit(self.n_qubits, 1)
        self.input_params = [Parameter(f"inp_{i}") for i in range(self.n_qubits)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(self.n_qubits)]

        # Encode inputs with Ry gates
        for i, p in enumerate(self.input_params):
            self.qc.ry(p, i)

        # Apply a depth‑dependent sequence of Rx rotations as trainable weights
        for _ in range(self.depth):
            for i, p in enumerate(self.weight_params):
                self.qc.rx(p, i)

        # Simple entangling pattern: a linear chain of CNOTs
        for i in range(self.n_qubits - 1):
            self.qc.cx(i, i + 1)

        # Observable: Pauli‑Y on the first qubit
        self.observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.tanh(self.hidden1(x))
        h2 = F.tanh(self.hidden2(h1))

        if h2.size(1)!= self.n_qubits:
            raise ValueError(f"Expected hidden size {self.n_qubits} for quantum layer, got {h2.size(1)}")

        param_dict = {p: h2[:, i] for i, p in enumerate(self.input_params)}
        for i, p in enumerate(self.weight_params):
            param_dict[p] = h2[:, i]

        q_out = self.qnn(param_dict)
        q_out = q_out.squeeze(-1)
        return self.out(q_out)

__all__ = ["HybridEstimatorQNN"]
