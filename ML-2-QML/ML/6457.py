import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Hybrid classifier that can operate in classical or quantum mode.
    Provides a consistent API for building a feed‑forward network or a
    variational quantum circuit, and for evaluating either an RBF or a
    quantum kernel.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 use_quantum: bool = False,
                 kernel_type: str = "rbf",
                 gamma: float = 1.0,
                 device: str = "cpu"):
        self.num_features = num_features
        self.depth = depth
        self.use_quantum = use_quantum
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.device = torch.device(device)

        # Build classical model
        self.classical_model, self.enc_cls, self.weights_cls, self.obs_cls = self.build_classifier_circuit(
            num_features, depth
        )

        if use_quantum:
            # Build quantum circuit
            self.quantum_circuit, self.enc_q, self.weights_q, self.obs_q = self.build_classifier_circuit(
                num_features, depth, quantum=True
            )

    def build_classifier_circuit(self,
                                 num_features: int,
                                 depth: int,
                                 quantum: bool = False) -> Tuple[nn.Module, Iterable, Iterable, List]:
        """
        Construct either a classical feed‑forward network or a quantum
        variational circuit.  Returns the model, an encoding list,
        a weight-size list, and a list of observables.
        """
        if not quantum:
            layers: List[nn.Module] = []
            in_dim = num_features
            encoding = list(range(num_features))
            weight_sizes = []
            for _ in range(depth):
                linear = nn.Linear(in_dim, num_features)
                layers.append(linear)
                layers.append(nn.ReLU())
                weight_sizes.append(linear.weight.numel() + linear.bias.numel())
                in_dim = num_features
            head = nn.Linear(in_dim, 2)
            layers.append(head)
            weight_sizes.append(head.weight.numel() + head.bias.numel())
            network = nn.Sequential(*layers).to(self.device)
            observables = list(range(2))
            return network, encoding, weight_sizes, observables

        # Quantum construction
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        encoding = ParameterVector("x", num_features)
        weights = ParameterVector("theta", num_features * depth)

        circuit = QuantumCircuit(num_features)
        for idx, qubit in enumerate(range(num_features)):
            circuit.rx(encoding[idx], qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_features):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_features - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_features - i - 1))
                       for i in range(num_features)]
        return circuit, list(encoding), list(weights), observables

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Classical RBF kernel."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Quantum kernel using the circuit defined in build_classifier_circuit."""
        from qiskit import Aer, execute
        backend = Aer.get_backend("statevector_simulator")

        # Encode x
        circuit_x = self.quantum_circuit.copy()
        param_map = {param: val.item() for param, val in zip(self.enc_q, x)}
        circuit_x.bind_parameters(param_map)
        job_x = execute(circuit_x, backend)
        state_x = job_x.result().get_statevector(circuit_x)

        # Encode y
        circuit_y = self.quantum_circuit.copy()
        param_map_y = {param: val.item() for param, val in zip(self.enc_q, y)}
        circuit_y.bind_parameters(param_map_y)
        job_y = execute(circuit_y, backend)
        state_y = job_y.result().get_statevector(circuit_y)

        overlap = np.abs(np.vdot(state_x, state_y)) ** 2
        return torch.tensor(overlap, device=self.device)

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        kernel_fn = self._rbf_kernel if self.kernel_type == "rbf" else self._quantum_kernel
        return np.array([[kernel_fn(x, y).item() for y in b] for x in a])

    def predict_classical(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical network."""
        X = X.to(self.device)
        with torch.no_grad():
            return self.classical_model(X)

    def predict_quantum(self, X: torch.Tensor) -> torch.Tensor:
        """Expectation value of observables for each sample."""
        if not self.use_quantum:
            raise RuntimeError("Quantum mode not enabled.")
        from qiskit import Aer, execute
        backend = Aer.get_backend("statevector_simulator")
        results = []
        for x in X:
            circuit = self.quantum_circuit.copy()
            param_map = {param: val.item() for param, val in zip(self.enc_q, x)}
            circuit.bind_parameters(param_map)
            job = execute(circuit, backend)
            state = job.result().get_statevector(circuit)
            exp_vals = [np.real(np.vdot(state, op.to_matrix().dot(state))) for op in self.obs_q]
            results.append(exp_vals)
        return torch.tensor(results, device=self.device)
