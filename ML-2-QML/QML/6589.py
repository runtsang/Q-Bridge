import numpy as np
import torch
from torch.optim import Adam
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNNQNN(n_features: int = 8, n_qubits: int = 8, n_layers: int = 3) -> EstimatorQNN:
    """
    Constructs a Qiskit EstimatorQNN that mirrors the classical encoder
    but replaces the fully‑connected layers with a variational quantum ansatz.
    """
    algorithm_globals.random_seed = 12345

    # Feature map: simple product of Hadamards and Rz rotations
    feature_map = QuantumCircuit(n_features)
    for i in range(n_features):
        feature_map.h(i)
        feature_map.rz(ParameterVector(f"φ{i}")[0], i)

    # Ansatz: alternating CNOT‑RY‑RZ blocks
    ansatz = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for i in range(0, n_qubits, 2):
            ansatz.cx(i, i + 1)
            ansatz.ry(ParameterVector(f"θ{layer}_{i}")[0], i)
            ansatz.rz(ParameterVector(f"θ{layer}_{i+1}")[0], i + 1)

    # Full circuit
    circuit = QuantumCircuit(n_features + n_qubits)
    circuit.compose(feature_map, range(n_features), inplace=True)
    circuit.compose(ansatz, range(n_features, n_features + n_qubits), inplace=True)

    # Observable: measure Z on the first qubit of the ansatz
    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    return qnn

def train_qcnn(
    qnn: EstimatorQNN,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 200
) -> EstimatorQNN:
    """
    Simple training loop for the EstimatorQNN.
    Uses a stochastic gradient descent optimizer (Adam) and binary cross‑entropy loss.
    """
    optimizer = Adam(qnn.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    X_np = X.numpy()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass with gradients
        preds, grads = qnn.predict(X_np, gradient=True)

        # Convert to torch tensors
        preds_t = torch.tensor(preds, dtype=torch.float32, requires_grad=True)
        loss = loss_fn(preds_t, y.float())

        # Backpropagate the loss
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return qnn
