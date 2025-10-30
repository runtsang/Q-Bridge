import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import StronglyEntanglingLayers
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

class EstimatorQNN(QiskitEstimatorQNN):
    """
    A quantum neural network using a 3‑qubit strongly entangling ansatz.
    The circuit is parameterised by both input and weight parameters,
    and the estimator is a StatevectorEstimator for exact gradients.
    """

    def __init__(self, input_dim: int = 3, depth: int = 2):
        # Build variational circuit
        input_params = [Parameter(f"input_{i}") for i in range(input_dim)]
        weight_params = [Parameter(f"w_{i}") for i in range(input_dim * depth * 3)]

        # Construct ansatz
        circuit = QuantumCircuit(input_dim)

        # Encode inputs as Rx rotations
        for i, p in enumerate(input_params):
            circuit.rx(p, i)

        # Add strongly entangling layers
        layer = StronglyEntanglingLayers(num_qubits=input_dim, reps=depth, insert_barriers=False)
        circuit.append(layer, range(input_dim))

        # Bind weight parameters to the layer's parameters
        bound_circuit = circuit.copy()
        bound_circuit.bind_parameters({p: w for p, w in zip(layer.parameters, weight_params)})

        # Observables: sum of Z on each qubit
        observables = [SparsePauliOp.from_list([("Z" * input_dim, 1)])]

        super().__init__(
            circuit=bound_circuit,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=Estimator(),
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 8,
    ) -> None:
        """
        Simple training loop using parameter‑shift gradients via PyTorch.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                xb, yb = X[perm[i : i + batch_size]], y[perm[i : i + batch_size]]
                xb_t = torch.tensor(xb, dtype=torch.float32, requires_grad=True)
                yb_t = torch.tensor(yb, dtype=torch.float32)

                preds = self.forward(xb_t)
                loss = criterion(preds, yb_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
