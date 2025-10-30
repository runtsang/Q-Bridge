"""Quantum Convolutional Neural Network built with Qiskit Machine Learning.

Highlights:
- Feature map: ZZFeatureMap (high‑order correlations)
- Ansatz: RealAmplitudes with circular entanglement
- Optional noise model for realistic simulations
- Adam‑based training helper that uses parameter‑shift gradients
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import Adam

def QCNN(num_qubits: int = 8,
         backend: AerSimulator | None = None,
         noise_model=None) -> EstimatorQNN:
    """Construct a QCNN as an EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    backend : AerSimulator, optional
        Backend to run the estimator on; if None an AerSimulator is created.
    noise_model : NoiseModel, optional
        Noise model to attach to the simulator.
    """
    if backend is None:
        backend = AerSimulator()
    if noise_model:
        backend.set_options(noise_model=noise_model)

    estimator = Estimator(backend=backend)

    # Feature map: captures correlations between all qubits
    feature_map = ZZFeatureMap(num_qubits, reps=2, entanglement='circular')

    # Ansatz: parameterised circuit with entanglement
    ansatz = RealAmplitudes(num_qubits, reps=3, entanglement='circular')

    # Assemble full circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observable: single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observable=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

def train_qcnn(qnn: EstimatorQNN,
               X_train: np.ndarray,
               y_train: np.ndarray,
               epochs: int = 50,
               lr: float = 0.01,
               batch_size: int = 16) -> None:
    """Train the QCNN using Adam and parameter‑shift gradients.

    The training loop evaluates the mean‑squared error on mini‑batches and
    updates the circuit parameters via the Adam optimiser.
    """
    opt = Adam(lr)
    opt.set_weights(qnn.get_weights())

    for epoch in range(epochs):
        permutation = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            idx = permutation[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            def loss_fn(weights):
                qnn.set_weights(weights)
                preds = qnn.predict(X_batch)
                return np.mean((preds - y_batch)**2)

            opt.step(loss_fn)
            epoch_loss += loss_fn(opt.get_weights())

        avg_loss = epoch_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

__all__ = ["QCNN", "train_qcnn"]
