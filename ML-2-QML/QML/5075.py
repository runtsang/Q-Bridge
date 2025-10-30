import torch
import numpy as np
import networkx as nx
from typing import Iterable, Sequence, List, Tuple, Optional
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN:
    """
    Quantum counterpart of the classical EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the encoding circuit.
    depth : int
        Number of variational layers.
    mode : str, optional
        ``"regression"`` or ``"classification"``.  The default is
        ``"regression"``.
    use_kernel : bool, optional
        If True, a quantum kernel is available via ``kernel_matrix``.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 mode: str = "regression",
                 use_kernel: bool = False):
        self.num_qubits = num_qubits
        self.depth = depth
        self.mode = mode
        self.use_kernel = use_kernel
        self.current_weights: Optional[np.ndarray] = None

        if mode == "regression":
            self.circuit, self.input_params, self.weight_params, self.observable = self._build_regression_circuit()
        else:
            self.circuit, self.input_params, self.weight_params, self.observables = self._build_classifier_circuit()

        if use_kernel:
            self.kernel = self._build_kernel()

        self.backend = Aer.get_backend('statevector_simulator')
        self.estimator = StatevectorEstimator()

    def set_weights(self, weights: Sequence[float]) -> None:
        """Set the variational parameters used in the circuit."""
        self.current_weights = np.array(weights)

    def _apply_params(self, circuit: QuantumCircuit, input_vals: np.ndarray) -> QuantumCircuit:
        binder = dict(zip(self.input_params, input_vals))
        if self.current_weights is not None:
            binder.update(dict(zip(self.weight_params, self.current_weights)))
        return circuit.assign_parameters(binder, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the model on a batch of inputs.

        For regression the Y expectation of the last qubit is returned.
        For classification the predicted class (0 or 1) is returned.
        """
        batch = x.shape[0]
        outputs = []
        for i in range(batch):
            inp = x[i].detach().cpu().numpy()
            circ = self.circuit.copy()
            circ = self._apply_params(circ, inp)
            state = Statevector.from_instruction(circ)
            if self.mode == "regression":
                exp = state.expectation_value(self.observable)
                outputs.append(torch.tensor(exp, dtype=torch.float32))
            else:
                probs = state.probabilities_dict()
                # probabilities for |0> and |1> for each qubit
                probs_vec = []
                for qubit in range(self.num_qubits):
                    key0 = '0'*qubit + '0' + '0'*(self.num_qubits - qubit - 1)
                    key1 = '0'*qubit + '1' + '0'*(self.num_qubits - qubit - 1)
                    p0 = probs.get(key0, 0.0)
                    p1 = probs.get(key1, 0.0)
                    probs_vec.extend([p0, p1])
                class_prob = torch.tensor(probs_vec, dtype=torch.float32).sum(dim=0)
                pred = torch.argmax(class_prob, dim=0)
                outputs.append(pred.float())
        return torch.stack(outputs)

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the quantum kernel Gram matrix between two datasets.
        """
        if not self.use_kernel:
            raise RuntimeError("Kernel functionality not enabled.")
        return np.array([[self._kernel_pair(x.numpy(), y.numpy()) for y in b] for x in a])

    def _kernel_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        circ_x = self.circuit.copy()
        circ_y = self.circuit.copy()
        binder_x = dict(zip(self.input_params, x))
        binder_y = dict(zip(self.input_params, y))
        if self.current_weights is not None:
            binder_x.update(dict(zip(self.weight_params, self.current_weights)))
            binder_y.update(dict(zip(self.weight_params, self.current_weights)))
        circ_x = circ_x.assign_parameters(binder_x, inplace=True)
        circ_y = circ_y.assign_parameters(binder_y, inplace=True)
        state_x = Statevector.from_instruction(circ_x)
        state_y = Statevector.from_instruction(circ_y)
        return abs((state_x.dag() * state_y)[0, 0]) ** 2

    def fidelity_adjacency(self,
                           states: Sequence[Statevector],
                           threshold: float,
                           *,
                           secondary: Optional[float] = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph from state fidelities of quantum states.
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1:], start=i + 1):
                fid = abs((a.dag() * b)[0, 0]) ** 2
                if fid >= threshold:
                    G.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    G.add_edge(i, j, weight=secondary_weight)
        return G

    @staticmethod
    def _build_regression_circuit() -> Tuple[QuantumCircuit, List[Parameter], List[Parameter], SparsePauliOp]:
        """
        Construct a regression‑style variational circuit.
        """
        params = ParameterVector("x", 4)
        weights = ParameterVector("theta", 16)

        qc = QuantumCircuit(4)
        for p, qubit in zip(params, range(4)):
            qc.rx(p, qubit)

        idx = 0
        for _ in range(4):
            for qubit in range(4):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(3):
                qc.cz(qubit, qubit + 1)

        observable = SparsePauliOp.from_list([("Y" * 4, 1)])
        return qc, list(params), list(weights), observable

    @staticmethod
    def _build_classifier_circuit() -> Tuple[QuantumCircuit, List[Parameter], List[Parameter], List[SparsePauliOp]]:
        """
        Construct a classification‑style variational circuit.
        """
        params = ParameterVector("x", 4)
        weights = ParameterVector("theta", 16)

        qc = QuantumCircuit(4)
        for p, qubit in zip(params, range(4)):
            qc.rx(p, qubit)

        idx = 0
        for _ in range(4):
            for qubit in range(4):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(3):
                qc.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (4 - i - 1)) for i in range(4)]
        return qc, list(params), list(weights), observables

    @staticmethod
    def _build_kernel() -> None:
        """
        Placeholder for a quantum kernel implementation.  The actual kernel
        is computed on demand via ``_kernel_pair``.
        """
        return None

__all__ = ["EstimatorQNN"]
