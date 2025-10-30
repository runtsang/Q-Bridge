"""Extended QCNN quantum circuit with shared parameters and classical post‑processing."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
import torch
from torch import nn


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution sub‑circuit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling sub‑circuit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _layer(num_qubits: int,
           pairings: list[tuple[int, int]],
           param_prefix: str,
           circuit_fn) -> QuantumCircuit:
    """Generic layer builder for convolution or pooling."""
    qc = QuantumCircuit(num_qubits, name=f"{circuit_fn.__name__}_layer")
    params = ParameterVector(param_prefix, length=len(pairings) * 3)
    idx = 0
    for (q1, q2) in pairings:
        sub = circuit_fn(params[idx: idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc


class QCNNExtendedQNN:
    """
    Hybrid quantum‑classical QCNN with:
      • Configurable number of convolution + pooling layers.
      • Shared parameters per layer type (convolution vs pooling).
      • Classical dense head after measurement.
      • Optional choice of backend (simulator or real device).
    """
    def __init__(self,
                 num_qubits: int = 8,
                 conv_layers: int = 3,
                 pool_layers: int = 3,
                 backend: str = "statevector_simulator",
                 dropout: float = 0.1,
                 seed: int | None = None):
        self.num_qubits = num_qubits
        self.backend = backend
        self.seed = seed

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits)
        feature_params = self.feature_map.parameters

        # Ansatz construction
        qc = QuantumCircuit(num_qubits)
        qc.compose(self.feature_map, range(num_qubits), inplace=True)

        # Define pairings for each layer (alternating)
        pairings = [(i, i + 1) for i in range(0, num_qubits, 2)]
        for layer in range(conv_layers):
            qc.compose(_layer(num_qubits,
                              pairings,
                              f"c{layer}",
                              _conv_circuit), range(num_qubits), inplace=True)
            qc.compose(_layer(num_qubits,
                              pairings,
                              f"p{layer}",
                              _pool_circuit), range(num_qubits), inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator
        estimator = Estimator(method=self.backend)

        # Quantum neural network
        self.qnn = EstimatorQNN(circuit=qc.decompose(),
                                observables=observable,
                                input_params=feature_params,
                                weight_params=qc.parameters,
                                estimator=estimator)

        # Classical head
        self.classical_head = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(8, 1)
        )

    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Forward pass: quantum measurement → classical dense head → sigmoid output.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        # Ensure shape (n_samples, features)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        qout = self.qnn.predict(x)  # shape (n_samples, 1)
        qout_torch = torch.from_numpy(qout).float()
        return torch.sigmoid(self.classical_head(qout_torch))

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int = 50,
              lr: float = 0.01) -> None:
        """
        Simple training loop using COBYLA optimizer for the quantum part
        and Adam for the classical head.
        """
        optimizer = COBYLA(maxiter=epochs * 10)
        opt_cl = torch.optim.Adam(self.classical_head.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            # Quantum forward
            qout = self.qnn.predict(X_train)
            qout_torch = torch.from_numpy(qout).float()
            # Classical head
            logits = self.classical_head(qout_torch)
            loss = loss_fn(logits.squeeze(), torch.from_numpy(y_train).float())

            # Backprop for classical part
            opt_cl.zero_grad()
            loss.backward()
            opt_cl.step()

            # Update quantum parameters via COBYLA
            def objective(params):
                self.qnn.weight_params = params
                qout = self.qnn.predict(X_train)
                qout_torch = torch.from_numpy(qout).float()
                logits = self.classical_head(qout_torch)
                loss = loss_fn(logits.squeeze(), torch.from_numpy(y_train).float())
                return loss.item()

            opt_cl.zero_grad()
            opt_cl.step()
            # COBYLA step
            current_params = self.qnn.weight_params
            new_params = optimizer.minimize(objective, current_params)
            self.qnn.weight_params = new_params

            if epoch % 10 == 0:
                print(f"Epoch {epoch} – Loss: {loss.item():.4f}")

def QCNNExtendedQNNFactory(**kwargs) -> QCNNExtendedQNN:
    """Convenience factory mirroring the classical QCNNExtended."""
    return QCNNExtendedQNN(**kwargs)

__all__ = ["QCNNExtendedQNN", "QCNNExtendedQNNFactory"]
